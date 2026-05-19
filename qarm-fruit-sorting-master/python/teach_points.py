"""
Real-time teach-pendant jog tool for the QArm with live camera view.

Display: single OpenCV window showing the D415 color stream plus a status
overlay with pose, gripper, step, saved-count, and the recent log line.

Focus that window to drive the arm (key codes via cv2.waitKeyEx):
  Arrows  : +X / -X  (Up/Down)   +Y / -Y  (Right/Left)
  r / f   : +Z / -Z
  q / e   : wrist - / +
  SPACE   : toggle gripper
  j       : toggle JOINT mode (per-joint jog — overrides q/e/r/f mapping)
              in JOINT mode:  q/a J0   w/s J1   e/d J2   r/f J3
              (rot_step applies; SPACE/+/-/h/p/n/l/x/m/g/1-9 still work)
  + / -   : double / halve jog step
  h       : slow home (5 s)
  p       : print current pose to terminal
  l       : view saved points (scrollable, read only; ESC to close)
  n       : save current pose (prompts for label in the terminal)
  x       : delete a saved point (prompts in the terminal)
  ESC     : write teach_points.json and exit
  Ctrl+C in terminal : quit WITHOUT saving

Defaults: 5 mm linear step, 5 deg wrist step.
Camera: opens the D415 via the Quanser SDK. If the camera fails to open
a black display is used and jog still works.
"""

import json
import os
import sys
import time
import threading
import traceback
from datetime import datetime

import numpy as np
import cv2

from qarm_driver import QArmDriver
from qarm_kinematics import (
    forward_kinematics, inverse_kinematics, JOINT_LIMITS
)

try:
    from camera import QArmCamera
except Exception as _cam_exc:
    QArmCamera = None
    _cam_import_error = _cam_exc


POINTS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "teach_points.json")
)
LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs")
)


from trace_logger import TraceLogger


DEFAULT_LIN_STEP = 0.005            # 5 mm
DEFAULT_ROT_STEP = np.deg2rad(1.5)  # 1.5 deg (wrist in Cart mode + all joints in Joint mode)
INTERP_HZ = 100.0

# Gripper safety: never fully close or fully open the gripper.
#   1.0 slams the jaws against the table/object  -> close overload (-1289)
#   0.0 drives the servo past its physical open endstop and holds it
#       there during the rest of the interpolation  -> open overload (-1289)
# Use safer clamps on both ends.
GRIP_CLOSE_CMD = 0.9
GRIP_OPEN_CMD  = 0.2
# If the end-effector is below this Z (m), warn before closing.
Z_LOW_WARN = 0.01
WINDOW_NAME = "QArm Teach Pendant — focus here to jog"

# cv2.waitKeyEx arrow key codes on Windows
KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_LEFT = 2424832
KEY_RIGHT = 2555904


class CameraThread:
    """Background thread for continuous camera frame capture."""

    def __init__(self, cam):
        self.cam = cam
        self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            try:
                color, _ = self.cam.read()
                with self._lock:
                    self._frame = color.copy()
            except Exception:
                pass
            time.sleep(0.016)  # ~60 fps

    def get_frame(self):
        with self._lock:
            return self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


def load_points():
    if os.path.exists(POINTS_FILE):
        try:
            with open(POINTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            print(f"[warn] {POINTS_FILE} corrupt, starting empty")
    return {}


def save_points(points):
    with open(POINTS_FILE, "w") as f:
        json.dump(points, f, indent=2)
    print(f"[ok] wrote {len(points)} points to {POINTS_FILE}")


def pose_from_joints(phi):
    # gamma in this project's IK is theta4 == phi4 (wrist), not a composite.
    pos, _ = forward_kinematics(phi)
    gamma = float(phi[3])
    return np.array(pos, dtype=float), gamma


def interp_move(q, phi_from, phi_to, grip_from, grip_to, seconds,
                logger=None, tag="INTERP", display_cb=None):
    n = max(2, int(seconds * INTERP_HZ))
    dt = 1.0 / INTERP_HZ
    if logger:
        logger.log(tag + "_BEGIN",
                   phi_from=phi_from, phi_to=phi_to,
                   grip_from=grip_from, grip_to=grip_to,
                   seconds=seconds, steps=n)
    for i in range(n + 1):
        s = i / n
        s_smooth = 3 * s * s - 2 * s * s * s
        phi = phi_from + s_smooth * (phi_to - phi_from)
        g = grip_from + s_smooth * (grip_to - grip_from)
        try:
            q.set_joints_and_gripper(phi, g)
        except Exception as ex:
            if logger:
                logger.log("HIL_ERROR", where="interp_move",
                           step=i, phi=phi, g=g, err=repr(ex))
            raise
        # Every ~10th step, read actual pose and log drift
        if logger and (i % max(1, n // 10) == 0):
            try:
                j_act, g_act = q.read_all()
                drift = float(np.max(np.abs(np.array(j_act) - phi)))
                logger.log(tag + "_STEP",
                           i=i, phi_cmd=phi, g_cmd=g,
                           phi_act=j_act, g_act=g_act,
                           drift_max=drift)
            except Exception as ex:
                logger.log("HIL_ERROR", where="read_all",
                           err=repr(ex))
        # Refresh live camera view every ~3 steps (~30 ms ≈ 33 fps)
        if display_cb and (i % 3 == 0):
            display_cb(phi, g)
        time.sleep(dt)
    if logger:
        logger.log(tag + "_END")


def open_camera_or_none():
    if QArmCamera is None:
        print(f"[warn] no camera module: {_cam_import_error}")
        return None
    try:
        cam = QArmCamera()
        cam.open()
        return cam
    except Exception as e:
        print(f"[warn] camera open failed: {e}  (running without live view)")
        return None


def _put(display, text, y, scale=0.55, color=(255, 255, 255), thick=2):
    cv2.putText(display, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(display, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def render_goto_panel(display, labels):
    """Top-right panel listing all saved points numbered 1..N.
    First 9 are direct 1-9 hotkeys; 10+ are accessed via g then t (type #)."""
    if not labels:
        return
    h, w = display.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    row_h = 20
    max_rows = max(1, (h - 80) // row_h)
    items = labels[:max_rows]
    truncated = len(labels) > max_rows

    max_text_w = 0
    for i, lbl in enumerate(items):
        (tw, _), _ = cv2.getTextSize(f"{i+1:2d}: {lbl}", font, scale, thick)
        max_text_w = max(max_text_w, tw)

    pad = 10
    header = f"POINTS  ({len(labels)})"
    (hw, _), _ = cv2.getTextSize(header, font, scale, thick)
    box_w = max(max_text_w, hw) + 2 * pad
    extra_h = row_h if truncated else 0
    box_h = 24 + len(items) * row_h + pad + extra_h
    x1 = w - 10
    x0 = x1 - box_w
    y0 = 10
    y1 = y0 + box_h

    overlay = display.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 1)

    cv2.putText(display, header, (x0 + pad, y0 + 18),
                font, scale, (0, 255, 255), thick)
    for i, lbl in enumerate(items):
        color = (255, 255, 255) if i < 9 else (180, 180, 180)
        cv2.putText(display, f"{i+1:2d}: {lbl}",
                    (x0 + pad, y0 + 38 + i * row_h),
                    font, scale, color, thick)
    if truncated:
        cv2.putText(display, f"... +{len(labels) - max_rows} more",
                    (x0 + pad, y0 + 38 + len(items) * row_h),
                    font, scale, (180, 180, 180), thick)


def render_overlay(display, phi_cmd, xyz, gamma, grip,
                   lin_step, rot_step, pts_count, log_line, recording,
                   goto_labels=None):
    h, w = display.shape[:2]

    _put(display, f"xyz=[{xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}] m", 28)
    _put(display, f"joints=[{np.degrees(phi_cmd[0]):+6.1f} "
                  f"{np.degrees(phi_cmd[1]):+6.1f} "
                  f"{np.degrees(phi_cmd[2]):+6.1f} "
                  f"{np.degrees(phi_cmd[3]):+6.1f}] deg", 52)
    _put(display, f"grip={grip:.2f}   step=lin{lin_step*1000:.1f}mm/"
                  f"rot{np.degrees(rot_step):.0f}deg   saved={pts_count}", 76)

    legend = [
        "arrows: X/Y   r/f: Z   q/e: wrist   SPACE: grip",
        "1-9: goto saved point   g: goto menu (t=type #)   n: save",
        "m: modify point   z: test routine",
        "+/-: step   l: view saved   x: del   h: home   p: pose",
        "ESC: save JSON and exit",
    ]
    for i, line in enumerate(legend):
        _put(display, line, h - 20 - (len(legend) - 1 - i) * 22,
             scale=0.5, color=(0, 255, 255), thick=1)

    if log_line:
        _put(display, log_line, 100, scale=0.55, color=(0, 255, 0), thick=2)

    if goto_labels:
        render_goto_panel(display, goto_labels)

    if recording:
        cv2.circle(display, (w - 20, 20), 8, (0, 0, 255), -1)


MENU_PAGE_SIZE = 9  # items visible per page


def render_menu_box(display, prompt, labels, page=0, cursor=-1, hint=None):
    """Render a numbered menu over the camera view with pagination.
    page: current page (0-based). cursor: highlighted item index (-1=none).
    hint: override for footer hint text."""
    h, w = display.shape[:2]
    total = len(labels)
    total_pages = max(1, (total + MENU_PAGE_SIZE - 1) // MENU_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * MENU_PAGE_SIZE
    end = min(start + MENU_PAGE_SIZE, total)
    visible = labels[start:end]
    n_items = len(visible)

    box_w = min(640, int(w * 0.7))
    box_h = 80 + n_items * 32 + 40
    x0 = (w - box_w) // 2
    y0 = max(10, (h - box_h) // 2)

    overlay = display.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
    cv2.rectangle(display, (x0, y0), (x0 + box_w, y0 + box_h),
                  (0, 255, 255), 2)

    title = prompt
    if total_pages > 1:
        title += f"  (page {page+1}/{total_pages})"
    cv2.putText(display, title, (x0 + 14, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for i, label in enumerate(visible):
        y = y0 + 60 + i * 32
        global_idx = start + i
        digit = i + 1
        line = f"{digit}: {label}"
        if global_idx == cursor:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)
        cv2.putText(display, line, (x0 + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    if hint is None:
        hint = "1-9: select   arrows: scroll   ESC: cancel"
    cv2.putText(display, hint,
                (x0 + 14, y0 + box_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def render_input_box(display, prompt, buffer, blink_on):
    """Render a modal textbox overlay at the center of the window."""
    h, w = display.shape[:2]
    box_w = min(640, int(w * 0.7))
    box_h = 120
    x0 = (w - box_w) // 2
    y0 = (h - box_h) // 2

    # Semi-transparent dark background
    overlay = display.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)

    # Border
    cv2.rectangle(display, (x0, y0), (x0 + box_w, y0 + box_h),
                  (0, 255, 255), 2)

    cv2.putText(display, prompt, (x0 + 14, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Text input line with cursor
    text = buffer
    cursor = "_" if blink_on else " "
    line = text + cursor
    cv2.putText(display, line, (x0 + 14, y0 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(display, "Enter: confirm    ESC: cancel",
                (x0 + 14, y0 + 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def modal_menu(window_name, get_frame_fn, phi_cmd, xyz, gamma, grip_cmd,
               lin_step, rot_step, pts_count, prompt, labels,
               read_only=False):
    """Show a paginated menu over the camera view. Returns the selected
    label string, or "" on ESC. When read_only=True, digit keys are
    ignored and only ESC exits (pure viewer)."""
    page = 0
    total_pages = max(1, (len(labels) + MENU_PAGE_SIZE - 1) // MENU_PAGE_SIZE)
    while True:
        frame = get_frame_fn()
        render_overlay(frame, phi_cmd, xyz, gamma, grip_cmd,
                       lin_step, rot_step, pts_count, "", False)
        render_menu_box(frame, prompt, labels, page=page)
        cv2.imshow(window_name, frame)
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        if key == 27:
            return ""
        # Arrow keys / PgUp / PgDn for page navigation
        if key == KEY_RIGHT or key == KEY_DOWN:
            page = min(page + 1, total_pages - 1)
        elif key == KEY_LEFT or key == KEY_UP:
            page = max(page - 1, 0)
        elif not read_only and ord("1") <= key <= ord("9"):
            idx = (page * MENU_PAGE_SIZE) + (key - ord("1"))
            if idx < len(labels):
                return labels[idx]


def modal_input(window_name, get_frame_fn, phi_cmd, xyz, gamma, grip_cmd,
                lin_step, rot_step, pts_count, prompt):
    """Show a modal textbox overlaid on the live camera view and return the
    typed string. ESC cancels (returns ""). Enter confirms.
    While the modal is up, the arm hold-pose is maintained externally.
    """
    buf = ""
    while True:
        frame = get_frame_fn()
        render_overlay(frame, phi_cmd, xyz, gamma, grip_cmd,
                       lin_step, rot_step, pts_count, "", False)
        blink_on = int(time.time() * 2) % 2 == 0
        render_input_box(frame, prompt, buf, blink_on)
        cv2.imshow(window_name, frame)
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        if key == 27:              # ESC
            return ""
        if key in (13, 10):        # Enter / LF
            return buf.strip()
        if key == 8:               # Backspace
            buf = buf[:-1]
            continue
        if 32 <= key < 127:        # printable ASCII
            buf += chr(key)
            continue


def modal_goto_select(window_name, get_frame_fn, phi_cmd, xyz, gamma, grip_cmd,
                      lin_step, rot_step, pts_count, labels):
    """Goto menu: paginated list of all saved points plus a textbox for
    typing a point number directly (needed for # >= 10 since digit keys
    only cover 1-9). Returns the selected label or '' on cancel."""
    page = 0
    total_pages = max(1, (len(labels) + MENU_PAGE_SIZE - 1) // MENU_PAGE_SIZE)
    hint = "1-9: select   t: type #   arrows: page   ESC: cancel"
    prompt = f"Go to point  ({len(labels)} saved)"
    while True:
        frame = get_frame_fn()
        render_overlay(frame, phi_cmd, xyz, gamma, grip_cmd,
                       lin_step, rot_step, pts_count, "", False)
        render_menu_box(frame, prompt, labels, page=page, hint=hint)
        cv2.imshow(window_name, frame)
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        if key == 27:
            return ""
        if key == KEY_RIGHT or key == KEY_DOWN:
            page = min(page + 1, total_pages - 1)
        elif key == KEY_LEFT or key == KEY_UP:
            page = max(page - 1, 0)
        elif ord("1") <= key <= ord("9"):
            idx = (page * MENU_PAGE_SIZE) + (key - ord("1"))
            if idx < len(labels):
                return labels[idx]
        elif key < 256 and chr(key).lower() == "t":
            num_str = modal_input(
                window_name, get_frame_fn,
                phi_cmd, xyz, gamma, grip_cmd,
                lin_step, rot_step, pts_count,
                f"Point # (1-{len(labels)}):")
            if not num_str:
                continue
            try:
                n = int(num_str)
            except ValueError:
                continue
            if 1 <= n <= len(labels):
                return labels[n - 1]


def main():
    points = load_points()
    print(f"Loaded {len(points)} existing points from {POINTS_FILE}")

    log_path = os.path.join(
        LOG_DIR, "robot_trace_" +
        datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    )
    trace = TraceLogger(log_path)
    print(f"Trace log: {log_path}")

    q = QArmDriver()
    q.connect()
    trace.log("QARM_CONNECTED")
    cam = open_camera_or_none()
    cam_thread = CameraThread(cam) if cam is not None else None
    trace.log("CAMERA", ok=(cam is not None))

    try:
        phi_cmd, grip_cmd = q.read_all()
        phi_cmd = np.array(phi_cmd, dtype=float)
        grip_cmd = float(grip_cmd)
        xyz, gamma = pose_from_joints(phi_cmd)
        trace.log("INITIAL_POSE", phi=phi_cmd, grip=grip_cmd, xyz=xyz)

        lin_step = DEFAULT_LIN_STEP
        rot_step = DEFAULT_ROT_STEP
        log_line = "ready"
        log_expiry = time.time() + 3.0
        joint_mode = False

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        def grab_frame():
            if cam_thread is not None:
                return cam_thread.get_frame()
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        def refresh_display(phi_now, grip_now):
            """Update the OpenCV window with live camera + overlay.
            Called from interp_move so the view stays alive during moves."""
            frame = grab_frame()
            xyz_now, _ = pose_from_joints(phi_now)
            render_overlay(frame, phi_now, xyz_now, gamma, grip_now,
                           lin_step, rot_step, len(points), log_line,
                           recording=False,
                           goto_labels=list(points.keys()))
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

        def set_gripper(target_grip, seconds=0.8):
            """Interpolate gripper without moving joints.
            After settling, reads the ACTUAL gripper position and stores
            it as grip_cmd. This way any subsequent joint+gripper write
            commands "stay where you already are" — zero motor torque
            needed — avoiding the QARM_OVERLOAD_GRIPPER fault (-1289).
            """
            nonlocal grip_cmd
            trace.log("GRIPPER_CMD", grip_from=grip_cmd, grip_to=target_grip,
                      xyz=xyz)
            interp_move(q, phi_cmd, phi_cmd, grip_cmd, target_grip,
                        seconds=seconds, logger=trace, tag="GRIPPER",
                        display_cb=refresh_display)
            # brief settle — just 3 writes, not 10
            for _ in range(3):
                try:
                    q.set_joints_and_gripper(phi_cmd, target_grip)
                except Exception as ex:
                    trace.log("HIL_ERROR", where="set_gripper_hold",
                              err=repr(ex))
                    raise
                time.sleep(0.05)
            # Read the ACTUAL gripper position — if the jaws stalled against
            # a fruit or limit, the real position is less than target. Save
            # THAT as the new grip_cmd so we stop pushing against the motor.
            try:
                _, actual_grip = q.read_all()
                grip_cmd = float(actual_grip)
                trace.log("GRIPPER_DONE",
                          target=target_grip, actual=actual_grip,
                          stall=(target_grip - actual_grip))
            except Exception as ex:
                trace.log("HIL_ERROR", where="set_gripper_readback",
                          err=repr(ex))
                grip_cmd = target_grip

        def run_test_routine():
            nonlocal log_line, log_expiry
            label_list = list(points.keys())
            if len(label_list) < 6:
                log_line = f"need 6 points for routine, have {len(label_list)}"
                log_expiry = time.time() + 3.0
                return
            trace.log("ROUTINE_START", labels=",".join(label_list[:6]))
            sequence = [
                ("goto", 1),
                ("goto", 0),
                ("goto", 5),
                ("grip", GRIP_CLOSE_CMD),
                ("dwell", 0.5),
                ("goto", 1),
                ("goto", 2),
                ("goto", 4),
                ("grip", GRIP_OPEN_CMD),
                ("dwell", 0.5),
                ("goto", 2),
                ("goto", 1),
            ]
            for step_idx, (op, arg) in enumerate(sequence):
                trace.log("ROUTINE_STEP", step=step_idx, op=op, arg=arg)
                if op == "goto":
                    print(f"\n[routine] goto position {arg + 1}: {label_list[arg]}")
                    goto_label(label_list[arg])
                elif op == "grip":
                    print(f"\n[routine] gripper -> {arg:.2f}")
                    set_gripper(arg)
                elif op == "dwell":
                    print(f"\n[routine] settling {arg:.1f}s...")
                    time.sleep(arg)
            trace.log("ROUTINE_DONE")
            log_line = "routine done"
            log_expiry = time.time() + 3.0

        def goto_label(label):
            nonlocal phi_cmd, grip_cmd, xyz, gamma, log_line, log_expiry
            if label not in points:
                log_line = f"no such point: {label}"
                log_expiry = time.time() + 3.0
                trace.log("GOTO_MISS", label=label)
                return
            pt = points[label]
            phi_target = np.array(pt["joints_rad"], dtype=float)
            grip_target = grip_cmd
            clipped = False
            for i in range(4):
                lo, hi = JOINT_LIMITS[i]
                if phi_target[i] < lo or phi_target[i] > hi:
                    clipped = True
                phi_target[i] = float(np.clip(phi_target[i], lo, hi))
            dist = float(np.max(np.abs(phi_target - phi_cmd)))
            duration = float(np.clip(1.5 + dist * 1.2, 1.5, 5.0))
            trace.log("GOTO", label=label,
                      phi_from=phi_cmd, phi_to=phi_target,
                      dist=dist, duration=duration, clipped=clipped)
            print(f"  going to {label}  duration={duration:.1f}s")
            interp_move(q, phi_cmd, phi_target,
                        grip_cmd, grip_target, seconds=duration,
                        logger=trace, tag="GOTO",
                        display_cb=refresh_display)
            for _ in range(20):
                try:
                    q.set_joints_and_gripper(phi_target, grip_target)
                except Exception as ex:
                    trace.log("HIL_ERROR", where="goto_settle",
                              err=repr(ex))
                    raise
                time.sleep(0.05)
            phi_cmd = phi_target
            grip_cmd = grip_target
            xyz, gamma = pose_from_joints(phi_cmd)
            try:
                j_act, g_act = q.read_all()
                err = float(np.max(np.abs(np.array(j_act) - phi_cmd)))
                trace.log("GOTO_END", label=label,
                          phi_cmd=phi_cmd, phi_act=j_act,
                          max_err=err, g_act=g_act)
            except Exception as ex:
                trace.log("HIL_ERROR", where="goto_verify", err=repr(ex))
            log_line = f"arrived at {label}"
            log_expiry = time.time() + 3.0

        while True:
            disp = grab_frame()
            if time.time() > log_expiry:
                log_line = ""
            if joint_mode:
                # Persistent banner overrides log_line while in joint mode
                log_line = ("JOINT mode  q/a J0  w/s J1  e/d J2  "
                            "r/f J3   (j: back to Cart)")
                log_expiry = time.time() + 1.0
            render_overlay(disp, phi_cmd, xyz, gamma, grip_cmd,
                           lin_step, rot_step, len(points), log_line,
                           recording=False,
                           goto_labels=list(points.keys()))

            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKeyEx(15)

            if key == -1:
                # idle — DO NOT resend commands. The QArm position-mode PID
                # holds the last setpoint on its own. Spamming the gripper
                # command into a stalled jaw position causes persistent
                # motor load and triggers QERR_QARM_OVERLOAD_GRIPPER (-1289).
                continue

            if key == 27:  # ESC
                save_points(points)
                return

            # --- Joint-mode toggle + joint-mode key interception --------
            # Works in either mode. 'j' flips Cartesian <-> Joint mode.
            if key < 256 and chr(key).lower() == "j":
                joint_mode = not joint_mode
                log_line = ("JOINT mode (q/a J0  w/s J1  e/d J2  r/f J3)"
                            if joint_mode else "Cartesian mode")
                log_expiry = time.time() + 3.0
                continue

            # In joint mode, q/a, w/s, e/d, r/f act on individual joints
            # instead of going through IK. Other keys (SPACE, h, p, n, +/-,
            # 1-9, l, g, m, x) pass through to the normal handlers below.
            if joint_mode and key < 256:
                c_jm = chr(key).lower()
                joint_idx = None
                delta = 0.0
                if c_jm == "q":   joint_idx, delta = 0, -rot_step
                elif c_jm == "a": joint_idx, delta = 0, +rot_step
                elif c_jm == "w": joint_idx, delta = 1, -rot_step
                elif c_jm == "s": joint_idx, delta = 1, +rot_step
                elif c_jm == "e": joint_idx, delta = 2, -rot_step
                elif c_jm == "d": joint_idx, delta = 2, +rot_step
                elif c_jm == "r": joint_idx, delta = 3, -rot_step
                elif c_jm == "f": joint_idx, delta = 3, +rot_step

                if joint_idx is not None:
                    phi_new = phi_cmd.copy()
                    target = phi_cmd[joint_idx] + delta
                    lo, hi = JOINT_LIMITS[joint_idx]
                    target_clipped = float(np.clip(target, lo, hi))
                    if abs(target - target_clipped) > 1e-9:
                        log_line = (f"J{joint_idx} at limit "
                                    f"({np.degrees(target_clipped):+.1f} deg)")
                        log_expiry = time.time() + 2.0
                    phi_new[joint_idx] = target_clipped
                    trace.log("JOINT_JOG", idx=joint_idx,
                              phi_from=phi_cmd, phi_to=phi_new)
                    interp_move(q, phi_cmd, phi_new, grip_cmd, grip_cmd,
                                seconds=0.3, logger=trace, tag="JOINT_JOG",
                                display_cb=refresh_display)
                    phi_cmd = phi_new
                    xyz, gamma = pose_from_joints(phi_cmd)
                    continue
                # Not a joint key — fall through so SPACE/h/p/n/etc work

            new_xyz = xyz.copy()
            new_gamma = gamma
            new_grip = grip_cmd
            moved = False

            if key == KEY_UP:
                new_xyz[0] += lin_step; moved = True
            elif key == KEY_DOWN:
                new_xyz[0] -= lin_step; moved = True
            elif key == KEY_RIGHT:
                new_xyz[1] += lin_step; moved = True
            elif key == KEY_LEFT:
                new_xyz[1] -= lin_step; moved = True
            elif key == 32:  # SPACE — use set_gripper so we read-back actual
                if grip_cmd > 0.50:
                    set_gripper(GRIP_OPEN_CMD)
                else:
                    if xyz[2] < Z_LOW_WARN:
                        log_line = (f"warn: Z={xyz[2]*1000:.0f}mm — "
                                    f"gripper capped to {GRIP_CLOSE_CMD}")
                        log_expiry = time.time() + 3.0
                    set_gripper(GRIP_CLOSE_CMD)
                continue  # set_gripper already handled everything
            elif ord("1") <= key <= ord("9"):
                # Direct goto by digit key (no 'g' needed)
                idx = key - ord("1")
                label_list = list(points.keys())
                if idx < len(label_list):
                    goto_label(label_list[idx])
                else:
                    log_line = f"no point #{idx+1}"
                    log_expiry = time.time() + 2.0
            elif key < 256:
                c = chr(key).lower()
                if c == "r":
                    new_xyz[2] += lin_step; moved = True
                elif c == "f":
                    new_xyz[2] -= lin_step; moved = True
                elif c == "q":
                    new_gamma -= rot_step; moved = True
                elif c == "e":
                    new_gamma += rot_step; moved = True
                elif c == "+" or c == "=":
                    lin_step *= 2; rot_step *= 2
                    log_line = (f"step lin={lin_step*1000:.1f}mm "
                                f"rot={np.degrees(rot_step):.0f}deg")
                    log_expiry = time.time() + 3.0
                elif c == "-" or c == "_":
                    lin_step /= 2; rot_step /= 2
                    log_line = (f"step lin={lin_step*1000:.1f}mm "
                                f"rot={np.degrees(rot_step):.0f}deg")
                    log_expiry = time.time() + 3.0
                elif c == "z":
                    run_test_routine()
                elif c == "h":
                    print("\n  homing (5 s)...")
                    target = np.zeros(4)
                    # Keep the gripper where it is — don't force it to 0.0
                    # (which would push the servo against the open endstop).
                    safe_grip = max(grip_cmd, GRIP_OPEN_CMD)
                    interp_move(q, phi_cmd, target, grip_cmd, safe_grip,
                                seconds=5.0, logger=trace, tag="HOME",
                                display_cb=refresh_display)
                    for _ in range(40):
                        q.set_joints_and_gripper(target, safe_grip)
                        time.sleep(0.05)
                    phi_cmd = target.copy()
                    # read back actual gripper so grip_cmd is harmless
                    try:
                        _, grip_cmd = q.read_all()
                        grip_cmd = float(grip_cmd)
                    except Exception:
                        grip_cmd = safe_grip
                    xyz, gamma = pose_from_joints(phi_cmd)
                    log_line = "homed"
                    log_expiry = time.time() + 3.0
                elif c == "p":
                    phi_now, grip_now = q.read_all()
                    xyz_now, gamma_now = pose_from_joints(phi_now)
                    print()
                    print(f"  actual joints deg: {np.degrees(phi_now).round(2)}")
                    print(f"  cmd    joints deg: {np.degrees(phi_cmd).round(2)}")
                    print(f"  xyz(m): {xyz_now.round(3)}  "
                          f"gamma(deg): {np.degrees(gamma_now):.1f}  "
                          f"gripper: {grip_now:.2f}")
                elif c == "l":
                    if not points:
                        log_line = "no points saved yet"
                        log_expiry = time.time() + 2.0
                        continue
                    modal_menu(
                        WINDOW_NAME, grab_frame,
                        phi_cmd, xyz, gamma, grip_cmd,
                        lin_step, rot_step, len(points),
                        "Saved points (read only)",
                        list(points.keys()),
                        read_only=True,
                    )
                elif c == "n":
                    label = modal_input(
                        WINDOW_NAME, grab_frame,
                        phi_cmd, xyz, gamma, grip_cmd,
                        lin_step, rot_step, len(points),
                        "Save point — label:",
                    )
                    if not label:
                        log_line = "cancelled"
                        log_expiry = time.time() + 2.0
                    else:
                        phi_act, grip_act = q.read_all()
                        xyz_act, gamma_act = pose_from_joints(phi_act)
                        points[label] = {
                            "label": label,
                            "joints_rad": [float(a) for a in phi_act],
                            "joints_deg": [float(np.degrees(a)) for a in phi_act],
                            "xyz_m": [float(a) for a in xyz_act],
                            "gamma_rad": float(gamma_act),
                            "gripper": float(grip_act),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                        log_line = f"saved {label}"
                        log_expiry = time.time() + 3.0
                        print(f"[saved] {label}  xyz={xyz_act.round(3)}")
                elif c == "g":
                    if not points:
                        log_line = "no points saved yet"
                        log_expiry = time.time() + 2.0
                        continue
                    label_list = list(points.keys())
                    label = modal_goto_select(
                        WINDOW_NAME, grab_frame,
                        phi_cmd, xyz, gamma, grip_cmd,
                        lin_step, rot_step, len(points),
                        label_list,
                    )
                    if label:
                        goto_label(label)
                    else:
                        log_line = "cancelled"
                        log_expiry = time.time() + 2.0
                elif c == "m":
                    if not points:
                        log_line = "no points saved yet"
                        log_expiry = time.time() + 2.0
                        continue
                    label_list = list(points.keys())
                    label = modal_menu(
                        WINDOW_NAME, grab_frame,
                        phi_cmd, xyz, gamma, grip_cmd,
                        lin_step, rot_step, len(points),
                        "Modify point — select to go there:",
                        label_list,
                    )
                    if label:
                        goto_label(label)
                        log_line = (f"at {label} — jog to new pos, "
                                    f"press m again to confirm")
                        log_expiry = time.time() + 60.0
                        # Wait for the user to jog, then press 'm'
                        # to overwrite with current position
                        while True:
                            disp = grab_frame()
                            if time.time() > log_expiry:
                                log_line = ""
                            render_overlay(
                                disp, phi_cmd, xyz, gamma, grip_cmd,
                                lin_step, rot_step, len(points),
                                log_line, recording=False,
                                goto_labels=list(points.keys()))
                            _put(disp, f"MODIFYING: {label}  "
                                 f"(m=save  ESC=cancel)", 128,
                                 scale=0.6, color=(0, 180, 255),
                                 thick=2)
                            cv2.imshow(WINDOW_NAME, disp)
                            mk = cv2.waitKeyEx(15)
                            if mk == -1:
                                continue
                            if mk == 27:  # ESC — cancel
                                log_line = "modify cancelled"
                                log_expiry = time.time() + 2.0
                                break

                            # Joint-mode toggle + interception inside modify
                            if mk < 256 and chr(mk).lower() == "j":
                                joint_mode = not joint_mode
                                log_line = (
                                    "JOINT mode (q/a J0 w/s J1 e/d J2 r/f J3)"
                                    if joint_mode else "Cartesian mode")
                                log_expiry = time.time() + 3.0
                                continue

                            if joint_mode and mk < 256:
                                c_jm = chr(mk).lower()
                                j_idx = None
                                j_delta = 0.0
                                if c_jm == "q":   j_idx, j_delta = 0, -rot_step
                                elif c_jm == "a": j_idx, j_delta = 0, +rot_step
                                elif c_jm == "w": j_idx, j_delta = 1, -rot_step
                                elif c_jm == "s": j_idx, j_delta = 1, +rot_step
                                elif c_jm == "e": j_idx, j_delta = 2, -rot_step
                                elif c_jm == "d": j_idx, j_delta = 2, +rot_step
                                elif c_jm == "r": j_idx, j_delta = 3, -rot_step
                                elif c_jm == "f": j_idx, j_delta = 3, +rot_step
                                if j_idx is not None:
                                    phi_new = phi_cmd.copy()
                                    target = phi_cmd[j_idx] + j_delta
                                    lo, hi = JOINT_LIMITS[j_idx]
                                    tc = float(np.clip(target, lo, hi))
                                    if abs(target - tc) > 1e-9:
                                        log_line = (f"J{j_idx} at limit "
                                                    f"({np.degrees(tc):+.1f} deg)")
                                        log_expiry = time.time() + 2.0
                                    phi_new[j_idx] = tc
                                    trace.log("JOINT_JOG_MODIFY",
                                              idx=j_idx,
                                              phi_from=phi_cmd, phi_to=phi_new)
                                    interp_move(
                                        q, phi_cmd, phi_new,
                                        grip_cmd, grip_cmd,
                                        seconds=0.3, logger=trace,
                                        tag="JOINT_JOG",
                                        display_cb=refresh_display)
                                    phi_cmd = phi_new
                                    xyz, gamma = pose_from_joints(phi_cmd)
                                    continue
                                # not a joint key — fall through to Cart handlers

                            # Jog controls work inside modify mode
                            m_xyz = xyz.copy()
                            m_gamma = gamma
                            m_moved = False
                            if mk == KEY_UP:
                                m_xyz[0] += lin_step; m_moved = True
                            elif mk == KEY_DOWN:
                                m_xyz[0] -= lin_step; m_moved = True
                            elif mk == KEY_RIGHT:
                                m_xyz[1] += lin_step; m_moved = True
                            elif mk == KEY_LEFT:
                                m_xyz[1] -= lin_step; m_moved = True
                            elif mk < 256:
                                mc = chr(mk).lower()
                                if mc == "r":
                                    m_xyz[2] += lin_step; m_moved = True
                                elif mc == "f":
                                    m_xyz[2] -= lin_step; m_moved = True
                                elif mc == "q":
                                    m_gamma -= rot_step; m_moved = True
                                elif mc == "e":
                                    m_gamma += rot_step; m_moved = True
                                elif mc == "+" or mc == "=":
                                    lin_step *= 2; rot_step *= 2
                                    log_line = (
                                        f"step lin={lin_step*1000:.1f}mm "
                                        f"rot={np.degrees(rot_step):.0f}deg")
                                    log_expiry = time.time() + 3.0
                                elif mc == "-" or mc == "_":
                                    lin_step /= 2; rot_step /= 2
                                    log_line = (
                                        f"step lin={lin_step*1000:.1f}mm "
                                        f"rot={np.degrees(rot_step):.0f}deg")
                                    log_expiry = time.time() + 3.0
                                elif mc == " ":
                                    if grip_cmd > 0.50:
                                        set_gripper(GRIP_OPEN_CMD)
                                    else:
                                        set_gripper(GRIP_CLOSE_CMD)
                                    continue
                                elif mc == "m":
                                    # Confirm: overwrite the point
                                    phi_act, grip_act = q.read_all()
                                    xyz_act, gamma_act = \
                                        pose_from_joints(phi_act)
                                    points[label] = {
                                        "label": label,
                                        "joints_rad": [float(a)
                                                       for a in phi_act],
                                        "joints_deg": [
                                            float(np.degrees(a))
                                            for a in phi_act],
                                        "xyz_m": [float(a)
                                                  for a in xyz_act],
                                        "gamma_rad": float(gamma_act),
                                        "gripper": float(grip_act),
                                        "timestamp":
                                            datetime.now().isoformat(
                                                timespec="seconds"),
                                    }
                                    log_line = f"modified {label}"
                                    log_expiry = time.time() + 3.0
                                    print(f"[modified] {label}  "
                                          f"xyz={xyz_act.round(3)}")
                                    break
                            if m_moved:
                                try:
                                    phi_new = inverse_kinematics(
                                        m_xyz, gamma=m_gamma)
                                except Exception as ex:
                                    log_line = f"IK fail: {ex}"
                                    log_expiry = time.time() + 2.0
                                    continue
                                phi_new = np.array(phi_new, dtype=float)
                                for i in range(4):
                                    lo, hi = JOINT_LIMITS[i]
                                    phi_new[i] = float(
                                        np.clip(phi_new[i], lo, hi))
                                interp_move(
                                    q, phi_cmd, phi_new,
                                    grip_cmd, grip_cmd,
                                    seconds=0.3, logger=trace,
                                    tag="JOG",
                                    display_cb=refresh_display)
                                phi_cmd = phi_new
                                xyz, gamma = pose_from_joints(phi_cmd)
                    else:
                        log_line = "cancelled"
                        log_expiry = time.time() + 2.0
                elif c == "x":
                    label = modal_input(
                        WINDOW_NAME, grab_frame,
                        phi_cmd, xyz, gamma, grip_cmd,
                        lin_step, rot_step, len(points),
                        "Delete point — label:",
                    )
                    if label in points:
                        del points[label]
                        log_line = f"deleted {label}"
                        log_expiry = time.time() + 3.0
                        print(f"[deleted] {label}")
                    elif label:
                        log_line = f"no such point: {label}"
                        log_expiry = time.time() + 3.0
                        print(f"no such point: {label}")

            if not moved:
                continue

            try:
                phi_new = inverse_kinematics(new_xyz, gamma=new_gamma)
            except Exception as ex:
                log_line = f"IK fail: {ex}"
                log_expiry = time.time() + 2.0
                trace.log("IK_FAIL", target=new_xyz, gamma=new_gamma,
                          err=repr(ex))
                continue

            phi_new = np.array(phi_new, dtype=float)
            out_of_limits = False
            for i in range(4):
                lo, hi = JOINT_LIMITS[i]
                if phi_new[i] < lo or phi_new[i] > hi:
                    out_of_limits = True
                phi_new[i] = float(np.clip(phi_new[i], lo, hi))
            if out_of_limits:
                log_line = "joint limit hit"
                log_expiry = time.time() + 2.0
                trace.log("JOINT_LIMIT", phi_new=phi_new)

            trace.log("JOG",
                      phi_from=phi_cmd, phi_to=phi_new,
                      grip_from=grip_cmd, grip_to=new_grip,
                      xyz_to=new_xyz)
            interp_move(q, phi_cmd, phi_new, grip_cmd, new_grip,
                        seconds=0.3, logger=trace, tag="JOG",
                        display_cb=refresh_display)
            phi_cmd = phi_new
            grip_cmd = new_grip
            xyz, gamma = pose_from_joints(phi_cmd)

    except KeyboardInterrupt:
        print("\n[ctrl+c] exiting WITHOUT saving")
        trace.log("INTERRUPTED")
    except Exception as ex:
        trace.log("EXCEPT", err=repr(ex), tb=traceback.format_exc())
        raise
    finally:
        trace.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if cam_thread is not None:
            try:
                cam_thread.stop()
            except Exception:
                pass
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
        try:
            q.card.close()
            q._connected = False
        except Exception:
            pass
        print("QArm disconnected (arm held at last commanded pose)")


if __name__ == "__main__":
    main()
