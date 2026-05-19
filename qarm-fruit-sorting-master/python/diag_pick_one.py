"""
Diagnostic: run detect_fruits once, then pick the highest-confidence
detection (or a user-selected one) to see whether the base-frame
coordinates really are pickable.

Each arm motion waits for ENTER before executing. Ctrl+C aborts at any
prompt; the arm holds at its last commanded pose.

Usage:
    py -3.13 python/diag_pick_one.py
    py -3.13 python/diag_pick_one.py --type tomato   # prefer this type
    py -3.13 python/diag_pick_one.py --idx 0         # pick Nth detection

Safety notes:
    - Approach hover at z=0.20 m, well above table.
    - Descend stops at max(0.03, fruit_top_z - 0.02) so we never drive
      below PICK_Z floor (0.02 m absolute).
    - Gripper closes to GRIP_CLOSE from sorting_controller (currently 0.90).
    - IK is solved BEFORE any motion so unreachable targets abort
      cleanly without moving the arm.
"""
from __future__ import annotations
import argparse
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
from qarm_kinematics import inverse_kinematics, JOINT_LIMITS
from sorting_controller import GRIP_OPEN, GRIP_CLOSE

APPROACH_Z = 0.20
DESCEND_OFFSET = 0.02     # grasp slightly below fruit top
PICK_Z_FLOOR = 0.03


def _capture(cam, deadline_s=10.0):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            c, _ = cam.read()
        except Exception:
            continue
        if c.mean() > 5:
            break
    else:
        raise RuntimeError("camera warm-up timeout")
    frames = []
    depth_latest = None
    for _ in range(5):
        c, d = cam.read()
        frames.append(c.copy())
        depth_latest = d.copy()
    color = np.median(np.stack(frames), axis=0).astype(np.uint8)
    intr = {k: float(cam.intrinsics[k]) for k in ("fx", "fy", "cx", "cy")}
    return color, depth_latest, intr


def _solve_or_abort(label: str, xyz: np.ndarray) -> np.ndarray:
    """Solve IK, return joints, or exit with clear message."""
    try:
        joints = inverse_kinematics(xyz, gamma=0.0)
    except Exception as ex:
        sys.exit(f"[abort] IK solve for {label} at {xyz} failed: {ex}")
    joints = np.asarray(joints, dtype=float)
    # Check joint limits
    for i, j in enumerate(joints):
        lo, hi = JOINT_LIMITS[i]
        if not (lo - 0.01 <= j <= hi + 0.01):
            sys.exit(f"[abort] {label} joint {i}={j:.3f} outside "
                     f"[{lo:.3f}, {hi:.3f}]")
    return joints


def _wait(prompt: str):
    try:
        input(f"  {prompt} (ENTER to go / Ctrl+C to abort)... ")
    except KeyboardInterrupt:
        sys.exit("[abort] user cancelled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=("banana", "tomato", "strawberry"),
                        default=None,
                        help="If set, prefer detections of this type.")
    parser.add_argument("--idx", type=int, default=None,
                        help="If set, pick detection at this index (0-based).")
    args = parser.parse_args()

    repo = os.path.dirname(_HERE)
    cal = SessionCal.load(os.path.join(repo, "session_cal.json"))
    print(f"  loaded session_cal  cam_h={cal.camera_height_above_table_m:.3f} m")

    from qarm_driver import QArmDriver
    from camera import QArmCamera
    from calibrate_closed_loop import slow_move_to_joints

    driver = QArmDriver()
    driver.connect()
    time.sleep(0.3)

    try:
        # 1. Survey -> capture -> detect
        tp_path = os.path.join(repo, "teach_points.json")
        with open(tp_path) as f:
            tp = json.load(f)
        grip = float(tp.get("survey1", {}).get("gripper", 0.10))
        print("\n  [1] moving to survey1 ...")
        slow_move_to_joints(driver, cal.survey_pose_joints_rad, grip)
        cam = QArmCamera()
        cam.open()
        try:
            color, depth, _ = _capture(cam)
        finally:
            cam.close()
        dets = detect_fruits(color, depth, cal)
        if not dets:
            sys.exit("[abort] no detections")
        print(f"\n  [2] detections:")
        for i, d in enumerate(dets):
            print(f"    [{i}] {d.fruit_type:10} conf={d.confidence:.2f}  "
                  f"base={tuple(round(float(x), 3) for x in d.center_base_m)}")

        # 2. Pick the target
        if args.idx is not None:
            if not (0 <= args.idx < len(dets)):
                sys.exit(f"[abort] --idx {args.idx} out of range")
            target = dets[args.idx]
        elif args.type is not None:
            matches = [d for d in dets if d.fruit_type == args.type]
            if not matches:
                sys.exit(f"[abort] no {args.type} detected")
            target = max(matches, key=lambda d: d.confidence)
        else:
            target = max(dets, key=lambda d: d.confidence)
        print(f"\n  [3] target: {target.fruit_type} at "
              f"{tuple(round(float(x), 3) for x in target.center_base_m)} "
              f"conf={target.confidence:.2f}")

        # 3. Pre-plan all motions with IK before moving
        fx, fy, fz = [float(v) for v in target.center_base_m]
        pick_z = max(PICK_Z_FLOOR, fz - DESCEND_OFFSET)

        hover_xyz = np.array([fx, fy, APPROACH_Z])
        pick_xyz = np.array([fx, fy, pick_z])
        lift_xyz = np.array([fx, fy, APPROACH_Z])

        hover_joints = _solve_or_abort("HOVER", hover_xyz)
        pick_joints = _solve_or_abort("PICK", pick_xyz)
        lift_joints = _solve_or_abort("LIFT", lift_xyz)
        print(f"\n  [4] IK solved for all 3 target poses. safe to proceed.")

        # 4. Execute
        print(f"\n  [5] HOVER at z={APPROACH_Z:.3f} ...")
        _wait("HOVER")
        slow_move_to_joints(driver, hover_joints, GRIP_OPEN)

        print(f"\n  [6] DESCEND to z={pick_z:.3f} ...")
        _wait("DESCEND")
        slow_move_to_joints(driver, pick_joints, GRIP_OPEN)

        print(f"\n  [7] CLOSE gripper to {GRIP_CLOSE} ...")
        _wait("CLOSE GRIPPER")
        # Gripper-only command: keep joints fixed, ramp gripper.
        joints_now, _ = driver.read_all()
        steps = 30
        for i in range(steps + 1):
            s = i / steps
            g = GRIP_OPEN + s * (GRIP_CLOSE - GRIP_OPEN)
            driver.set_joints_and_gripper(joints_now, g)
            time.sleep(0.03)
        time.sleep(0.5)
        _, g_act = driver.read_all()
        print(f"    gripper readback: {g_act:.3f}  (commanded {GRIP_CLOSE})")

        print(f"\n  [8] LIFT to z={APPROACH_Z:.3f} ...")
        _wait("LIFT")
        slow_move_to_joints(driver, lift_joints, GRIP_CLOSE)

        print("\n  [9] DONE — arm holds at lift pose, gripper still closed.")
        print("      Eyeball: did the gripper actually pick up the fruit?")
        print("      To release: driver.set_joints_and_gripper(joints, 0.15)")
        print("      To return home: run `py -3.13 python/teach_points.py`")

    finally:
        try: driver.card.close()
        except Exception: pass
        driver._connected = False


if __name__ == "__main__":
    main()
