"""
main_gui.py — Tkinter GUI for the QArm fruit-sorting system.

Layout:
  +---------------------+----------------------------+
  | live D415 video     | EMERGENCY STOP             |
  | (640 x 360)         |                            |
  |                     | Manual Teleop (XYZ +/-)    |
  |                     |   move to basket: B / T / S|
  |                     |                            |
  |                     | Gripper: open / close      |
  |                     | Camera: survey1 / refresh  |
  |                     |                            |
  |                     | Mode: Manual | Auto        |
  |                     | Auto: all / b / t / s      |
  |                     |                            |
  |                     | Status:  ...               |
  +---------------------+----------------------------+

Run:
  py -3.13 python/main_gui.py

Dependencies (beyond the existing project):
  - tkinter (stdlib on python.org Windows builds)
  - Pillow  (pip install pillow) — needed for cv2 -> Tk image conversion

Known limitations:
  - Emergency stop sets an abort flag and refreshes the joint command,
    but it CANNOT interrupt a slow_move_to_joints() that is already
    inside its 200-step interpolation loop. The arm finishes the current
    segment (~3-4 s worst case) before the abort takes effect.
  - The background camera reader pauses automatically during capture_fruits()
    to avoid concurrent camera reads (the Quanser Video3D driver is not
    thread-safe for concurrent reads).
  - No preflight; assumes session_cal.json is fresh and chess origin is
    correct. Run main_final.py first if you want the safety checks.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: this GUI requires Pillow.  pip install pillow", file=sys.stderr)
    raise

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from session_cal import SessionCal
from qarm_driver import QArmDriver
from camera import QArmCamera
from sorting_controller import (FruitSortingController,
                                  GRIP_OPEN, GRIP_CLOSE)
from qarm_kinematics import forward_kinematics, inverse_kinematics
from trace_logger import TraceLogger
import survey_capture as _survey_capture_module
import calibrate_closed_loop as _ccl_module
from picker_viewer import _pick_category

try:
    from voice_control import VoiceController
    _HAS_VOICE = True
except ImportError:
    _HAS_VOICE = False


_TYPE_COLORS = {
    "banana":     (0, 255, 255),    # yellow (BGR)
    "tomato":     (0, 0, 255),      # red
    "strawberry": (200, 100, 255),  # pink-magenta
}


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

TELEOP_STEP_M = 0.02            # metres per teleop button press
TELEOP_SECONDS = 0.6            # slow_move_to_joints duration for a teleop step
TELEOP_STEPS = 40               # interpolation steps for a teleop motion
BASKET_SECONDS = 2.0            # slow_move_to_joints duration for basket move
SURVEY_SECONDS = 2.5            # slow_move_to_joints duration for survey1 move

VIDEO_W = 640
VIDEO_H = 360
CAMERA_LOOP_HZ = 30
UI_REFRESH_HZ = 20
OVERLAY_LINGER_S = 5.0          # how long to hold the captured frame +
                                # detection bboxes on screen after a refresh
ESTOP_FREEZE_REPEAT = 30        # how many times to spam the freeze pose
ESTOP_FREEZE_INTERVAL = 0.01    # seconds between freeze repeats (=300 ms total)

REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_SESSION_CAL = os.path.join(REPO_ROOT, "session_cal.json")
DEFAULT_TEACH_POINTS = os.path.join(REPO_ROOT, "teach_points.json")
LOG_DIR = os.path.join(REPO_ROOT, "logs")


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class FruitSortingGUI:
    def __init__(self):
        # --- hardware + calibration ---
        if not os.path.exists(DEFAULT_SESSION_CAL):
            raise SystemExit(
                f"missing {DEFAULT_SESSION_CAL} — run "
                "calibrate_chessboard.py first.")
        self.session_cal = SessionCal.load(DEFAULT_SESSION_CAL)
        with open(DEFAULT_TEACH_POINTS) as f:
            self.teach_points = json.load(f)

        # --- trace log: one file per GUI session, written next to the
        #     teach_points trace logs (logs/gui_trace_<YYYYMMDD>_<HHMMSS>.log).
        #     Closed in _on_close so SESSION_END lands even on window-X exit.
        log_path = os.path.join(
            LOG_DIR,
            f"gui_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.trace = TraceLogger(log_path)
        self.trace.log("GUI_INIT",
                        cam_h=self.session_cal.camera_height_above_table_m,
                        chess_origin=self.session_cal.chess_origin_in_base_m,
                        homography_rms=self.session_cal.homography_reproj_rms_px)
        print(f"trace log: {log_path}")

        print("connecting to QArm + D415...")
        self.driver = QArmDriver()
        self.driver.connect()
        time.sleep(0.3)
        self.camera = QArmCamera()
        self.camera.open()
        self.controller = FruitSortingController(
            self.driver, camera=self.camera, logger=self.trace)

        # --- state ---
        self.mode = "manual"
        self.busy = False
        self.busy_lock = threading.Lock()
        self.abort_event = threading.Event()
        self._camera_paused = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._last_capture_frame = None
        self._last_detections: list = []
        self._last_capture_time = 0.0

        self._install_hooks()

        # --- UI ---
        self.root = tk.Tk()
        self.root.title("QArm Fruit Sorting — GUI")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._on_mode_change()  # set initial enable/disable

        # --- background threads ---
        self._stop_threads = threading.Event()
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True)
        self._camera_thread.start()

        # --- voice control (optional) ---
        if _HAS_VOICE:
            command_map = {
                "strawberry": lambda: self._auto_one("strawberry"),
                "banana":     lambda: self._auto_one("banana"),
                "tomato":     lambda: self._auto_one("tomato"),
                "all":        self._auto_all,
                "stop":       self._emergency_stop,
                "home":       self._goto_survey1,
                "refresh":    self._refresh_detect,
                "survey":     self._goto_survey1,
            }
            try:
                self.voice = VoiceController(
                    self.root, command_map,
                    status_label=self.voice_label)
            except Exception as ex:
                print(f"voice control disabled: {ex}")
                self.voice = None
        else:
            self.voice = None

        # schedule periodic video updates
        self.root.after(int(1000 / UI_REFRESH_HZ), self._update_video)

    # -----------------------------------------------------------------
    # Runtime monkey-patches (installed in __init__, cleaned up on close)
    # -----------------------------------------------------------------

    def _install_hooks(self):
        self._orig_capture_fruits = _survey_capture_module.capture_fruits
        self._orig_slow_move = _ccl_module.slow_move_to_joints
        gui = self

        def _capture_hook(driver, camera, session_cal):
            gui._camera_paused.set()
            try:
                dets, diag = gui._orig_capture_fruits(driver, camera, session_cal)
            finally:
                gui._camera_paused.clear()
            gui._on_capture(diag.get("color_frame"), dets)
            return dets, diag

        def _slow_move_hook(q, target, grip, seconds=3.0, steps=200):
            gui._safe_move_to_joints(target, grip,
                                     seconds=seconds, steps=steps)

        _survey_capture_module.capture_fruits = _capture_hook
        _ccl_module.slow_move_to_joints = _slow_move_hook

    def _uninstall_hooks(self):
        _survey_capture_module.capture_fruits = self._orig_capture_fruits
        _ccl_module.slow_move_to_joints = self._orig_slow_move

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def _build_ui(self):
        root = self.root

        # left: video only
        left = ttk.Frame(root, padding=4)
        left.grid(row=0, column=0, sticky="n")
        # IMPORTANT: do NOT set tk.Label width/height in character units —
        # Label without an image interprets them as char cols/rows
        # (640 chars ≈ 3840 px), which pushes everything else off-screen.
        # Instead pre-allocate a blank PhotoImage of the right pixel
        # size so the Label's initial layout is correct.
        self._blank_image = ImageTk.PhotoImage(Image.new(
            "RGB", (VIDEO_W, VIDEO_H), color="black"))
        self.video_label = tk.Label(left, image=self._blank_image,
                                       bg="black")
        self.video_label.image = self._blank_image  # GC anchor
        self.video_label.pack()

        self.voice_label = tk.Label(left, text="", font=("Arial", 10),
                                    bg="black", anchor="w")
        self.voice_label.pack(fill="x")

        # right: controls
        right = ttk.Frame(root, padding=8)
        right.grid(row=0, column=1, sticky="n")

        # ----- emergency stop -----
        tk.Button(right, text="EMERGENCY STOP", bg="#cc0000", fg="white",
                  font=("Arial", 14, "bold"), height=2,
                  command=self._emergency_stop).pack(fill="x", pady=(0, 8))

        # ----- mode toggle -----
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=4)
        mode_row = ttk.Frame(right); mode_row.pack(fill="x")
        ttk.Label(mode_row, text="Mode:").pack(side="left")
        self.mode_var = tk.StringVar(value="manual")
        ttk.Radiobutton(mode_row, text="Manual", variable=self.mode_var,
                         value="manual", command=self._on_mode_change
                         ).pack(side="left", padx=4)
        ttk.Radiobutton(mode_row, text="Auto", variable=self.mode_var,
                         value="auto", command=self._on_mode_change
                         ).pack(side="left", padx=4)

        # ----- manual teleop -----
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(right, text="Manual Teleop (base frame, "
                              f"{TELEOP_STEP_M*100:.0f} cm step)"
                  ).pack(anchor="w")
        self.manual_block = ttk.Frame(right); self.manual_block.pack(fill="x")

        # XYZ pad
        pad = ttk.Frame(self.manual_block); pad.pack(pady=2)
        self.manual_buttons = []

        def _mb(parent, text, r, c, fn):
            b = tk.Button(parent, text=text, width=6,
                          command=lambda: self._teleop(*fn))
            b.grid(row=r, column=c, padx=1, pady=1)
            self.manual_buttons.append(b)
            return b

        # row 0: +Y (forward / away from operator)
        _mb(pad, "Y+",  0, 1, (0, +TELEOP_STEP_M, 0))
        _mb(pad, "Z+",  0, 3, (0, 0, +TELEOP_STEP_M))
        # row 1: -X / center / +X
        _mb(pad, "X-",  1, 0, (-TELEOP_STEP_M, 0, 0))
        _mb(pad, "X+",  1, 2, (+TELEOP_STEP_M, 0, 0))
        # row 2: -Y / Z-
        _mb(pad, "Y-",  2, 1, (0, -TELEOP_STEP_M, 0))
        _mb(pad, "Z-",  2, 3, (0, 0, -TELEOP_STEP_M))

        # baskets row
        ttk.Label(self.manual_block, text="Move to basket:").pack(
            anchor="w", pady=(6, 0))
        baskets = ttk.Frame(self.manual_block); baskets.pack(fill="x")
        b1 = tk.Button(baskets, text="Strawberry", width=10,
                       command=lambda: self._move_to_basket("strawberry"))
        b1.pack(side="left", padx=2)
        b2 = tk.Button(baskets, text="Tomato", width=10,
                       command=lambda: self._move_to_basket("tomato"))
        b2.pack(side="left", padx=2)
        b3 = tk.Button(baskets, text="Banana", width=10,
                       command=lambda: self._move_to_basket("banana"))
        b3.pack(side="left", padx=2)
        self.manual_buttons.extend([b1, b2, b3])

        # gripper
        ttk.Label(self.manual_block, text="Gripper:").pack(
            anchor="w", pady=(6, 0))
        grip = ttk.Frame(self.manual_block); grip.pack(fill="x")
        bg1 = tk.Button(grip, text="Open",  width=10,
                        command=self._gripper_open)
        bg1.pack(side="left", padx=2)
        bg2 = tk.Button(grip, text="Close", width=10,
                        command=self._gripper_close)
        bg2.pack(side="left", padx=2)
        self.manual_buttons.extend([bg1, bg2])

        # camera ops
        ttk.Label(self.manual_block, text="Camera:").pack(
            anchor="w", pady=(6, 0))
        cam = ttk.Frame(self.manual_block); cam.pack(fill="x")
        bc1 = tk.Button(cam, text="Goto Survey1", width=12,
                        command=self._goto_survey1)
        bc1.pack(side="left", padx=2)
        bc2 = tk.Button(cam, text="Refresh detect", width=12,
                        command=self._refresh_detect)
        bc2.pack(side="left", padx=2)
        self.manual_buttons.extend([bc1, bc2])

        # ----- auto block -----
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(right, text="Auto Pick + Place").pack(anchor="w")
        self.auto_block = ttk.Frame(right); self.auto_block.pack(fill="x")
        self.auto_buttons = []
        ba = tk.Button(self.auto_block, text="ALL  (B → T → S)",
                       command=self._auto_all, height=2)
        ba.pack(fill="x", pady=2)
        bb = tk.Button(self.auto_block, text="Bananas only",
                       command=lambda: self._auto_one("banana"))
        bb.pack(fill="x", pady=1)
        bt = tk.Button(self.auto_block, text="Tomatoes only",
                       command=lambda: self._auto_one("tomato"))
        bt.pack(fill="x", pady=1)
        bs = tk.Button(self.auto_block, text="Strawberries only",
                       command=lambda: self._auto_one("strawberry"))
        bs.pack(fill="x", pady=1)
        self.auto_buttons.extend([ba, bb, bt, bs])

        # ----- status line -----
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status_var, wraplength=260,
                  foreground="#003300").pack(anchor="w")

    # -----------------------------------------------------------------
    # Mode handling
    # -----------------------------------------------------------------

    def _on_mode_change(self):
        self.mode = self.mode_var.get()
        manual_state = "normal" if self.mode == "manual" else "disabled"
        auto_state   = "normal" if self.mode == "auto"   else "disabled"
        for b in self.manual_buttons:
            b.configure(state=manual_state)
        for b in self.auto_buttons:
            b.configure(state=auto_state)
        self._set_status(f"Mode: {self.mode}")

    # -----------------------------------------------------------------
    # Camera background reader + UI video update
    # -----------------------------------------------------------------

    def _camera_loop(self):
        period = 1.0 / CAMERA_LOOP_HZ
        while not self._stop_threads.is_set():
            if self._camera_paused.is_set():
                time.sleep(period)
                continue
            try:
                color, _ = self.camera.read()
                with self._frame_lock:
                    self._latest_frame = color
            except Exception:
                pass
            time.sleep(period)

    def _on_capture(self, frame, dets):
        """Hook called from the monkey-patched capture_fruits — stores the
        captured frame + detection list for the video overlay."""
        with self._frame_lock:
            self._last_capture_frame = (None if frame is None
                                          else frame.copy())
            self._last_detections = list(dets)
            self._last_capture_time = time.time()

    def _draw_overlay(self, frame, dets):
        """Mutates `frame` in place — bbox + label + center dot per det."""
        for d in dets:
            color = _TYPE_COLORS.get(d.fruit_type, (255, 255, 255))
            x, y, w, h = d.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{d.fruit_type} {d.confidence:.2f}"
            cv2.putText(frame, label, (x, max(18, y - 6)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                         cv2.LINE_AA)
            cv2.circle(frame, tuple(int(v) for v in d.center_px),
                        4, color, -1)

    def _update_video(self):
        # Prefer the most recent captured frame (with bboxes) if it's
        # still within OVERLAY_LINGER_S; otherwise fall back to the
        # live camera feed without any overlay.
        with self._frame_lock:
            now = time.time()
            if (self._last_capture_frame is not None
                    and now - self._last_capture_time < OVERLAY_LINGER_S):
                frame = self._last_capture_frame.copy()
                dets = list(self._last_detections)
            else:
                frame = (None if self._latest_frame is None
                         else self._latest_frame.copy())
                dets = []
        if frame is not None and frame.size > 0:
            if dets:
                self._draw_overlay(frame, dets)
            small = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.configure(image=tkimg)
            self.video_label.image = tkimg  # keep ref so GC doesn't kill it
        if not self._stop_threads.is_set():
            self.root.after(int(1000 / UI_REFRESH_HZ), self._update_video)

# -----------------------------------------------------------------
    # Status helper (thread-safe via root.after)
    # -----------------------------------------------------------------

    def _set_status(self, msg: str):
        # Tk widget access from worker threads must be marshalled.
        self.root.after(0, lambda: self.status_var.set(msg))

    # -----------------------------------------------------------------
    # Worker dispatch (one long action at a time)
    # -----------------------------------------------------------------

    def _busy_guard(self, fn, label):
        with self.busy_lock:
            if self.busy:
                self._set_status(
                    f"Busy ({label} requested while another action runs)")
                return
            self.busy = True

        def worker():
            try:
                fn()
            except Exception as ex:
                self._set_status(f"{label} error: {ex}")
                import traceback; traceback.print_exc()
            finally:
                with self.busy_lock:
                    self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    # -----------------------------------------------------------------
    # Emergency stop
    # -----------------------------------------------------------------

    def _emergency_stop(self):
        """Set the abort flag, then spam the current measured pose back to
        the driver fast enough to win against any in-flight motion loop
        for a few hundred ms. Worker threads (teleop / basket / survey
        moves and the FSM tick observer) all check abort_event and bail
        within one step or one FSM tick of the flag being set, so the
        spam covers the brief window before they exit."""
        self.abort_event.set()
        try:
            joints, grip = self.driver.read_all()
            joints = np.asarray(joints, dtype=float)
            grip = float(grip)
            for _ in range(ESTOP_FREEZE_REPEAT):
                try:
                    self.driver.set_joints_and_gripper(joints, grip)
                except Exception:
                    pass
                time.sleep(ESTOP_FREEZE_INTERVAL)
            self._set_status(
                f"EMERGENCY STOP — arm frozen at "
                f"joints {np.round(joints, 3).tolist()}. "
                "Press another button to resume.")
        except Exception as ex:
            self._set_status(f"E-STOP partial: {ex}")

    def _safe_move_to_joints(self, target, grip,
                                seconds=3.0, steps=200):
        """Joint-space smoothstep interp + tail hammer-loop, but checks
        abort_event at every step so an E-STOP aborts within one tick
        (≤ seconds/steps ≈ 15 ms by default). Drop-in replacement for
        slow_move_to_joints inside the GUI; the underlying
        slow_move_to_joints in calibrate_closed_loop.py is left alone
        so non-GUI callers (preflight / capture_fruits / diag scripts)
        keep their original behaviour."""
        cj, cg = self.driver.read_all()
        cj = np.asarray(cj, dtype=float)
        cg = float(cg)
        target = np.asarray(target, dtype=float)
        grip = float(grip)
        for i in range(1, steps + 1):
            if self.abort_event.is_set():
                return False
            a = i / steps
            s = 3 * a * a - 2 * a * a * a
            self.driver.set_joints_and_gripper(
                cj + s * (target - cj),
                cg + s * (grip - cg))
            time.sleep(seconds / steps)
        # tail hammer-loop (matches slow_move_to_joints' last second)
        for _ in range(20):
            if self.abort_event.is_set():
                return False
            self.driver.set_joints_and_gripper(target, grip)
            time.sleep(0.05)
        return True

    # -----------------------------------------------------------------
    # Manual actions
    # -----------------------------------------------------------------

    def _teleop(self, dx, dy, dz):
        self.abort_event.clear()
        def fn():
            joints, grip = self.driver.read_all()
            ee_pos, _ = forward_kinematics(np.asarray(joints, dtype=float))
            new_pos = ee_pos + np.array([dx, dy, dz], dtype=float)
            try:
                tgt = inverse_kinematics(new_pos, gamma=0.0)
            except Exception as ex:
                self._set_status(
                    f"IK failed for {new_pos.round(3)}: {ex}")
                return
            ok = self._safe_move_to_joints(tgt, grip,
                                              seconds=TELEOP_SECONDS,
                                              steps=TELEOP_STEPS)
            if ok:
                self._set_status(f"Moved to ee = {new_pos.round(3)}")
            else:
                self._set_status("Teleop aborted (E-STOP)")
        self._busy_guard(fn, f"teleop ({dx:+.2f},{dy:+.2f},{dz:+.2f})")

    def _move_to_basket(self, fruit_type):
        self.abort_event.clear()
        def fn():
            basket_pos = self.controller.BASKETS.get(fruit_type)
            if basket_pos is None:
                self._set_status(f"no basket for {fruit_type}")
                return
            joints, grip = self.driver.read_all()
            try:
                tgt = inverse_kinematics(basket_pos.copy(), gamma=0.0)
            except Exception as ex:
                self._set_status(f"basket IK fail: {ex}")
                return
            ok = self._safe_move_to_joints(tgt, grip, seconds=BASKET_SECONDS)
            if ok:
                self._set_status(f"At {fruit_type} basket")
            else:
                self._set_status("Basket move aborted (E-STOP)")
        self._busy_guard(fn, f"to-basket {fruit_type}")

    def _gripper_open(self):
        self.abort_event.clear()
        def fn():
            actual = self.controller.set_gripper_ramp(GRIP_OPEN)
            self._set_status(f"Gripper open (held = {actual:.2f})")
        self._busy_guard(fn, "gripper open")

    def _gripper_close(self):
        self.abort_event.clear()
        def fn():
            actual = self.controller.set_gripper_ramp(GRIP_CLOSE)
            self._set_status(f"Gripper closed (held = {actual:.2f})")
        self._busy_guard(fn, "gripper close")

    def _goto_survey1(self):
        self.abort_event.clear()
        def fn():
            grip = float(self.teach_points.get("survey1", {}).get(
                "gripper", 0.10))
            ok = self._safe_move_to_joints(
                self.session_cal.survey_pose_joints_rad,
                grip, seconds=SURVEY_SECONDS)
            if ok:
                self._set_status("At survey1")
            else:
                self._set_status("Survey1 move aborted (E-STOP)")
        self._busy_guard(fn, "goto survey1")

    def _refresh_detect(self):
        self.abort_event.clear()
        def fn():
            dets, diag = _survey_capture_module.capture_fruits(
                self.driver, self.camera, self.session_cal)
            counts = {}
            for d in dets:
                counts[d.fruit_type] = counts.get(d.fruit_type, 0) + 1
            self._set_status(
                f"Detected {len(dets)} fruits "
                f"({', '.join(f'{k}={v}' for k, v in counts.items()) or '-'})")
        self._busy_guard(fn, "refresh detect")

    # -----------------------------------------------------------------
    # Auto actions
    # -----------------------------------------------------------------

    def _install_abort_observer(self):
        """Wire abort_event into the controller's tick_observer so that
        any FSM-driven motion (pick_single inside _pick_category) bails
        within one FSM tick of E-STOP. NOTE: sorting_controller's
        _drive_until_done catches observer exceptions and only logs
        a [warn], so raising alone is NOT enough to actually stop the
        FSM. We also force-set controller.state = State.DONE which
        makes the loop's `while self.state != State.DONE` check exit
        on the next iteration."""
        from sorting_controller import State as _State

        def _abort_observer():
            if self.abort_event.is_set():
                self.controller.state = _State.DONE
                raise RuntimeError("ABORT (E-STOP)")
        self.controller.tick_observer = _abort_observer

    def _clear_abort_observer(self):
        self.controller.tick_observer = None

    def _auto_one(self, fruit_type):
        self.abort_event.clear()
        def fn():
            self._set_status(f"Auto: picking all {fruit_type}…")
            self._install_abort_observer()
            try:
                n = _pick_category(self.driver, self.camera, self.session_cal,
                                    self.controller, fruit_type,
                                    lambda: self.abort_event.is_set())
            finally:
                self._clear_abort_observer()
            self._set_status(
                f"Auto: {fruit_type} done — {n} picked"
                + (" (aborted)" if self.abort_event.is_set() else ""))
        self._busy_guard(fn, f"auto {fruit_type}")

    def _auto_all(self):
        self.abort_event.clear()
        def fn():
            total = 0
            self._install_abort_observer()
            try:
                for ftype in ("banana", "tomato", "strawberry"):
                    if self.abort_event.is_set():
                        break
                    self._set_status(f"Auto-all: {ftype}…")
                    n = _pick_category(self.driver, self.camera,
                                        self.session_cal, self.controller,
                                        ftype,
                                        lambda: self.abort_event.is_set())
                    total += n
            finally:
                self._clear_abort_observer()
            self._set_status(
                f"Auto-all done — {total} picked"
                + (" (aborted)" if self.abort_event.is_set() else ""))
        self._busy_guard(fn, "auto all")

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def _on_close(self):
        self._stop_threads.set()
        self.abort_event.set()
        self._uninstall_hooks()
        if self.voice:
            try: self.voice.stop()
            except Exception: pass
        try: self.trace.close()
        except Exception: pass
        try: self.camera.close()
        except Exception: pass
        try: self.driver.card.close()
        except Exception: pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  QArm Fruit Sorting — GUI")
    print("=" * 60)
    gui = FruitSortingGUI()
    print("UI running.  Close the window to exit.")
    gui.run()


if __name__ == "__main__":
    main()
