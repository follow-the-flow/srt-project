"""
Diagnostic: move arm to survey1, capture one D415 frame, run detect_fruits,
and save an annotated PNG plus a terminal report.

Usage:
    py -3.13 python/diag_detector.py

Requires session_cal.json (produced by calibrate_chessboard.py) in repo root.
Does NOT require Phase 1 jog-touch — everything the detector needs is already
in session_cal.

Purpose: before D3 picker_viewer is built, validate that the D2 detector's
default HSV ranges actually find fruit in lab lighting. If the output
detections match what's on the table (quantity + type + plausible XYZ),
D2 is field-validated and D3 can proceed. If not, run hsv_tuner (not yet
written) or tweak HSV_RANGES by hand.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from session_cal import SessionCal
from fruit_detector import detect_fruits


_TYPE_COLORS = {
    "banana":     (0, 255, 255),   # yellow in BGR
    "tomato":     (0, 0, 255),     # red
    "strawberry": (200, 100, 255), # pink-magenta
}


def _warmup_and_capture(cam, warmup_timeout_s: float = 10.0):
    """Same pattern as calibrate_chessboard.py: poll cam.read() until the
    color frame's mean exceeds 5, then take 5 frames and median. Returns
    (color_median, depth_latest, intrinsics_dict)."""
    deadline = time.time() + warmup_timeout_s
    while time.time() < deadline:
        try:
            c, _ = cam.read()
        except Exception:
            continue
        if c.mean() > 5:
            break
    else:
        raise RuntimeError(
            "camera warm-up timeout - no valid frame after 10 s. "
            "Unplug/replug D415 or power-cycle the QArm.")
    color_frames = []
    depth_latest = None
    for _ in range(5):
        c, d = cam.read()
        color_frames.append(c.copy())
        depth_latest = d.copy()
    color = np.median(np.stack(color_frames), axis=0).astype(np.uint8)
    intr = {
        "fx": float(cam.intrinsics["fx"]),
        "fy": float(cam.intrinsics["fy"]),
        "cx": float(cam.intrinsics["cx"]),
        "cy": float(cam.intrinsics["cy"]),
    }
    return color, depth_latest, intr


def _annotate(bgr: np.ndarray, detections: list) -> np.ndarray:
    out = bgr.copy()
    for d in detections:
        color = _TYPE_COLORS.get(d.fruit_type, (255, 255, 255))
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{d.fruit_type} {d.confidence:.2f}"
        cv2.putText(out, label, (x, max(18, y - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.circle(out, d.center_px, 4, color, -1)
    return out


def main():
    repo = os.path.dirname(_HERE)
    cal_path = os.path.join(repo, "session_cal.json")
    if not os.path.exists(cal_path):
        sys.exit(f"missing {cal_path} - run calibrate_chessboard.py first")
    cal = SessionCal.load(cal_path)
    print(f"  loaded {cal_path}")
    print(f"    cam height:    {cal.camera_height_above_table_m:.3f} m")
    print(f"    chess origin:  {cal.chess_origin_in_base_m}")

    # Move arm to survey1. Uses calibrate_closed_loop.slow_move_to_joints
    # which is the existing helper; will be moved to motion_utils on D8.
    from qarm_driver import QArmDriver
    from camera import QArmCamera
    from calibrate_closed_loop import slow_move_to_joints

    driver = QArmDriver()
    driver.connect()
    time.sleep(0.3)

    try:
        tp_path = os.path.join(repo, "teach_points.json")
        with open(tp_path) as f:
            tp = json.load(f)
        grip = float(tp.get("survey1", {}).get("gripper", 0.10))
        print("\n  moving arm to survey1 ...")
        slow_move_to_joints(driver, cal.survey_pose_joints_rad, grip)

        cam = QArmCamera()
        cam.open()
        try:
            color, depth, intr = _warmup_and_capture(cam)
        finally:
            cam.close()
        print(f"  captured frame shape={color.shape} "
              f"depth_nonzero={(depth > 0).mean()*100:.0f}%")

        # Use the fresh intrinsics read this session (sometimes D415 reports
        # subtly different fx after a warm-up from what was in session_cal).
        cal_for_run = SessionCal(
            timestamp=cal.timestamp,
            chess_origin_in_base_m=cal.chess_origin_in_base_m,
            h_pixel_to_chess_mm=cal.h_pixel_to_chess_mm,
            survey_pose_joints_rad=cal.survey_pose_joints_rad,
            chess_pattern=cal.chess_pattern,
            d415_intrinsics=intr,
            homography_reproj_rms_px=cal.homography_reproj_rms_px,
            camera_height_above_table_m=cal.camera_height_above_table_m,
            image_size=cal.image_size,
            cam_extrinsics_survey1=cal.cam_extrinsics_survey1,
        )
        detections = detect_fruits(color, depth, cal_for_run)
    finally:
        try: driver.card.close()
        except Exception: pass
        driver._connected = False

    logs_dir = os.path.join(repo, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    raw_path = os.path.join(logs_dir, "diag_detector_raw.png")
    ann_path = os.path.join(logs_dir, "diag_detector_annotated.png")
    cv2.imwrite(raw_path, color)
    cv2.imwrite(ann_path, _annotate(color, detections))

    print(f"\n  {len(detections)} detection(s):")
    if detections:
        print(f"  {'type':<12} {'px':>12} {'base_m':>30} "
              f"{'conf':>6} {'area':>7}")
        for d in detections:
            px = f"({d.center_px[0]},{d.center_px[1]})"
            base = (f"({d.center_base_m[0]:+.3f},"
                    f"{d.center_base_m[1]:+.3f},"
                    f"{d.center_base_m[2]:+.3f})")
            print(f"  {d.fruit_type:<12} {px:>12} {base:>30} "
                  f"{d.confidence:>6.2f} {d.area_px:>7}")
    else:
        print("  (no fruit detected)")
    print(f"\n  raw frame:       {raw_path}")
    print(f"  annotated frame: {ann_path}")
    print(f"\n  open the annotated PNG to eyeball which blobs got picked up.")


if __name__ == "__main__":
    main()
