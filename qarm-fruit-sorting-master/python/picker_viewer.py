"""
Operator-facing picker UI: category-batch autonomous loop.

See docs/superpowers/specs/2026-04-22-overhead-vision-pipeline-design.md §3.5.

Event loop lives in run_picker_loop(). Everything above it is pure and
unit-tested.
"""
from __future__ import annotations
import os
import sys
import threading
import time
import traceback

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fruit_detector import Detection

_TYPE_COLORS = {
    "banana":     (0, 255, 255),    # yellow (BGR)
    "tomato":     (0, 0, 255),      # red
    "strawberry": (200, 100, 255),  # pink-magenta
}
_WARN_COLOR = (0, 180, 220)         # amber
_HUD_COLOR = (255, 255, 255)
_RESIDUAL_OK = (0, 200, 0)
_RESIDUAL_WARN = (0, 180, 220)
_RESIDUAL_BAD = (0, 0, 255)

def _filter_by_type(detections, fruit_type):
    """Keep only detections whose fruit_type matches."""
    return [d for d in detections if d.fruit_type == fruit_type]


def _residual_color(residual_mm):
    if residual_mm is None or residual_mm < 3.0:
        return _RESIDUAL_OK
    if residual_mm < 10.0:
        return _RESIDUAL_WARN
    return _RESIDUAL_BAD


def _annotate(color_bgr, detections, residual_mm, warnings):
    """Draw bboxes + labels + HUD on a copy of color_bgr."""
    out = color_bgr.copy()
    for d in detections:
        color = _TYPE_COLORS.get(d.fruit_type, (255, 255, 255))
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{d.fruit_type} {d.confidence:.2f}"
        cv2.putText(out, label, (x, max(18, y - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.circle(out, tuple(int(v) for v in d.center_px), 4, color, -1)
    hud = _hud_text(len(detections), residual_mm, "idle")
    cv2.putText(out, hud, (10, 28),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, _HUD_COLOR, 2, cv2.LINE_AA)
    # residual chip (top-right)
    if residual_mm is not None:
        rc = _residual_color(residual_mm)
        cv2.circle(out, (out.shape[1] - 30, 28), 10, rc, -1)
    for i, w in enumerate(warnings or []):
        cv2.putText(out, w, (10, 56 + i * 24),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WARN_COLOR, 1,
                     cv2.LINE_AA)
    return out


def _hud_text(n_fruits, residual_mm, mode):
    r = f"{residual_mm:.1f}" if residual_mm is not None else "n/a"
    return (f"{n_fruits} fruits  |  residual {r} mm  |  "
            f"mode: {mode}  |  b=banana  t=tomato  s=strawberry  "
            f"m=all  r=refresh  ESC=quit")


# ------------------------------------------------------------------------
# Live camera feed (used during arm motion to keep the picker window from
# freezing). The feed is the SOLE reader of QArmCamera while it runs
# because camera.read() shares internal buffers (camera.py:50-55, 168-174)
# and is not thread-safe. Callers MUST stop()+join the feed before any
# other code touches the camera.
# ------------------------------------------------------------------------

class _LiveFeed:
    def __init__(self, camera):
        self._cam = camera
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def _loop(self):
        while not self._stop.is_set():
            try:
                color, _ = self._cam.read()
                with self._lock:
                    self._latest = color
            except Exception:
                # Camera blip: keep last frame, retry next iteration.
                pass


def _make_render_observer(feed, window, controller, fps_limit=30.0):
    """Closure suitable for FruitSortingController.tick_observer.

    Reads the latest frame from `feed`, draws a status overlay derived
    from controller.state.name and controller.current_target, and shows
    it in `window`. Throttled to fps_limit so the FSM tick budget is
    not consumed by GUI work (cv2.imshow + waitKey costs ~1-3 ms; at
    100 Hz dt that would eat 10-30% of the loop). Uses time.monotonic()
    for the gate so a wall-clock NTP/DST jump cannot break the throttle."""
    last = {"t": 0.0}
    period = 1.0 / float(fps_limit)

    def _render():
        now = time.monotonic()
        if now - last["t"] < period:
            return
        last["t"] = now
        frame = feed.latest()
        if frame is None:
            return
        out = frame.copy()
        try:
            sname = controller.state.name
        except Exception:
            sname = "?"
        target = getattr(controller, "current_target", None)
        if target is not None:
            ftype = target.get("type", "?") if isinstance(target, dict) \
                else "?"
            pos = target.get("pos") if isinstance(target, dict) else None
            if pos is not None and len(pos) >= 3:
                label = (f"PICKING {ftype} @ "
                         f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                         f"state: {sname}")
            else:
                label = f"PICKING {ftype} | state: {sname}"
        else:
            label = f"state: {sname}"
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, out)
        cv2.waitKey(1)

    return _render


# ------------------------------------------------------------------------
# Pick dispatchers
# ------------------------------------------------------------------------

MAX_CATEGORY_PICKS = 20
RETRY_LIMIT_PER_TARGET = 2

_STRAWBERRY_CALYX_BIAS_M = 0.02   # empirical 2026-04-27 (re-added after
                                   # switching to widest-inscribed-disk
                                   # centroid): even with the centroid
                                   # already pulled into the wide body,
                                   # nudging another 2cm toward the calyx
                                   # lands the gripper better-centred on
                                   # the body. Banana/tomato unaffected.

_STRAWBERRY_X_BIAS_M = -0.030     # empirical 2026-04-27: strawberry picks
                                   # were landing too far from the base
                                   # (the global PICK_BIAS_X = +5cm in
                                   # the controller was over-correcting
                                   # for strawberries). Negative shifts
                                   # the target toward the base in the
                                   # base-frame x axis. Iterations -2cm
                                   # -> -3.5cm -> -2.5cm -> -3.0cm in
                                   # the same lab session (-3.5 over-
                                   # corrected, -2.5 went 1cm too far
                                   # away from base, -3.0 is the 0.5cm
                                   # midpoint). Net effective x bias
                                   # for strawberry = +5cm - 3.0cm
                                   # = +2.0cm. Banana keeps the full +5cm.

_TOMATO_X_BIAS_M = +0.020         # 2026-04-28 lab. Same sign convention as
                                   # strawberry above: positive shifts the
                                   # target away from the base. Net
                                   # effective x bias for tomato = +5cm +
                                   # 2cm = +7cm. Iterations the same
                                   # session: +1cm (net +6cm) -> -2cm
                                   # (net +3cm) -> +3cm (net +8cm, 1cm
                                   # too far) -> +2cm (net +7cm — current).


def _pick_one(controller, detection, camera=None, window=None) -> bool:
    """Dispatch one synchronous pick. Returns True on success.

    If both `camera` and `window` are given, runs a live D415 feed
    overlay during the arm motion via controller.tick_observer.
    The feed is the sole reader of `camera` for its lifetime; on
    return (success or exception) the feed is stopped + joined and
    the previous tick_observer is restored.
    """
    target = np.asarray(detection.center_base_m, dtype=float).copy()
    if detection.fruit_type == "strawberry":
        if detection.calyx_dir_base_unit is not None:
            target[:2] += _STRAWBERRY_CALYX_BIAS_M * np.asarray(
                detection.calyx_dir_base_unit, dtype=float)
        target[0] += _STRAWBERRY_X_BIAS_M
        print(f"  [picker] picking {detection.fruit_type} at "
              f"{target.round(3)} (widest-point "
              f"{detection.center_base_m.round(3)} + "
              f"{_STRAWBERRY_CALYX_BIAS_M*100:.1f}cm calyx bias + "
              f"{_STRAWBERRY_X_BIAS_M*100:+.1f}cm x bias, "
              f"conf={detection.confidence:.2f})")
    elif detection.fruit_type == "tomato":
        target[0] += _TOMATO_X_BIAS_M
        print(f"  [picker] picking {detection.fruit_type} at "
              f"{target.round(3)} (centroid "
              f"{detection.center_base_m.round(3)} + "
              f"{_TOMATO_X_BIAS_M*100:+.1f}cm x bias, "
              f"conf={detection.confidence:.2f})")
    else:
        print(f"  [picker] picking {detection.fruit_type} at "
              f"{target.round(3)} "
              f"(conf={detection.confidence:.2f})")
    feed = None
    prev_observer = getattr(controller, "tick_observer", None)
    try:
        if camera is not None and window is not None:
            feed = _LiveFeed(camera)
            feed.start()
            controller.tick_observer = _make_render_observer(
                feed, window, controller)
        return bool(controller.pick_single(target, detection.fruit_type))
    except Exception as ex:
        print(f"  [picker] pick_single raised: {ex}")
        traceback.print_exc()
        return False
    finally:
        if feed is not None:
            controller.tick_observer = prev_observer
            feed.stop()


def _nearest_to_home(detections, home_xy_m):
    if not detections:
        return None
    hx, hy = float(home_xy_m[0]), float(home_xy_m[1])
    return min(detections, key=lambda d: (
        (d.center_base_m[0] - hx) ** 2 +
        (d.center_base_m[1] - hy) ** 2))


def _pixel_key(detection, bucket_px=5):
    cx, cy = detection.center_px
    return (int(cx) // bucket_px * bucket_px,
            int(cy) // bucket_px * bucket_px)


def _pick_category(driver, camera, session_cal, controller,
                     fruit_type, should_abort_fn, window=None):
    """Category batch: re-capture, pick nearest-to-home, repeat until
    none left, limit hit, or abort signalled. Returns count picked.

    When `window` is given, _pick_one runs the live D415 feed overlay
    during each arm motion via controller.tick_observer."""
    from survey_capture import capture_fruits
    picks_done = 0
    retries = {}
    skipped_keys = set()
    while picks_done < MAX_CATEGORY_PICKS:
        # Service OpenCV events so ESC can abort between picks (spec §7.5).
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("  [picker] ESC pressed — aborting batch")
            return picks_done
        if should_abort_fn():
            print("  [picker] batch aborted")
            break
        try:
            dets, diag = capture_fruits(driver, camera, session_cal)
        except Exception as ex:
            print(f"  [picker] capture failed mid-batch: {ex}")
            break
        matches = [d for d in _filter_by_type(dets, fruit_type)
                   if _pixel_key(d) not in skipped_keys]
        if not matches:
            break
        target = _nearest_to_home(matches, controller.HOME_POS[:2])
        key = _pixel_key(target)
        success = _pick_one(controller, target,
                              camera=camera, window=window)
        if success:
            picks_done += 1
            retries.pop(key, None)
        else:
            retries[key] = retries.get(key, 0) + 1
            if retries[key] >= RETRY_LIMIT_PER_TARGET:
                print(f"  [picker] marking stuck target {key} as skipped")
                skipped_keys.add(key)
    return picks_done


# ------------------------------------------------------------------------
# Event loop
# ------------------------------------------------------------------------

def run_picker_loop(driver, camera, session_cal, controller) -> None:
    """Interactive OpenCV picker. Blocks until ESC.

    Parameters
    ----------
    driver     : QArmDriver (connected)
    camera     : QArmCamera (opened)
    session_cal: SessionCal
    controller : FruitSortingController
    """
    from survey_capture import capture_fruits
    window = "Picker"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    state = {"abort": False}

    def _refresh():
        dets, diag = capture_fruits(driver, camera, session_cal)
        frame = _annotate(diag["color_frame"], dets,
                            diag.get("chessboard_residual_mm"),
                            diag.get("warnings", []))
        cv2.imshow(window, frame)
        return dets, diag

    try:
        dets, diag = _refresh()
    except Exception as ex:
        print(f"  [picker] initial capture failed: {ex}")
        cv2.destroyWindow(window)
        return

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            print("  [picker] ESC — exiting")
            break
        if key in (ord('b'), ord('t'), ord('s')):
            ftype = {ord('b'): 'banana',
                       ord('t'): 'tomato',
                       ord('s'): 'strawberry'}[key]
            matches = _filter_by_type(dets, ftype)
            if not matches:
                print(f"  [picker] no {ftype} visible")
            else:
                print(f"  [picker] category batch: {ftype}")
                n = _pick_category(driver, camera, session_cal,
                                    controller, ftype,
                                    lambda: state["abort"],
                                    window=window)
                print(f"  [picker] {ftype} batch done: {n} picked")
                try:
                    dets, diag = _refresh()
                except Exception as ex:
                    print(f"  [picker] re-capture failed: {ex}")
        elif key == ord('m'):
            print("  [picker] multi-category sequence: "
                  "banana -> tomato -> strawberry")
            total = 0
            for ftype in ('banana', 'tomato', 'strawberry'):
                if (cv2.waitKey(1) & 0xFF) == 27 or state["abort"]:
                    print("  [picker] multi-category aborted between batches")
                    break
                print(f"  [picker] === {ftype} category ===")
                n = _pick_category(driver, camera, session_cal,
                                    controller, ftype,
                                    lambda: state["abort"],
                                    window=window)
                print(f"  [picker] {ftype} batch done: {n} picked")
                total += n
            print(f"  [picker] multi-category sequence done: "
                  f"{total} total picks")
            try:
                dets, diag = _refresh()
            except Exception as ex:
                print(f"  [picker] re-capture failed: {ex}")
        elif key == ord('r'):
            print("  [picker] refresh")
            try:
                dets, diag = _refresh()
            except Exception as ex:
                print(f"  [picker] re-capture failed: {ex}")

    cv2.destroyWindow(window)
