"""
Hover test — moves the arm to a safe position above each detected strawberry
without any contact. Validates calibration + motion pipeline in one pass.

Flow:
  1. Move to pickhome1, warmup camera, detect strawberries.
  2. For each detection, compute hover target (xy = fruit, z = fruit + 150mm,
     min 0.28 m above base).
  3. Safe motion: go up to z=0.35 waypoint, move laterally, descend to hover.
  4. Dwell 3 s, log FK tip, ascend, next fruit.
  5. After all fruits, return to pickhome1.

Safety:
  - Z is never commanded below SAFE_MIN_Z (well above 0.13 m table height).
  - Lateral motion always at z >= SAFE_TRANSIT_Z.
  - IK failures or joint-limit hits skip the fruit.
  - Gripper untouched throughout.
"""

import json
import os
import sys
import time
import numpy as np
import cv2

from camera import QArmCamera
from qarm_driver import QArmDriver
from qarm_kinematics import (forward_kinematics, inverse_kinematics,
                              JOINT_LIMITS)
import fruit_detector as fd
from fruit_detector import detection_depth_mm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS_FILE = os.path.join(REPO_ROOT, "teach_points.json")
CALIB_FILE = os.path.join(REPO_ROOT, "calibration.json")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

HOME_LABEL = "pickhome1"

# Safety envelope
SAFE_MIN_Z = 0.25          # never command below this (m)
SAFE_TRANSIT_Z = 0.35      # lateral moves happen at this altitude (m)
HOVER_DZ = 0.15            # target hover = fruit_z + HOVER_DZ
DWELL_SEC = 3.0

# Workspace clamp — fruits expected inside this box
WS_X = (0.15, 0.55)
WS_Y = (-0.35, 0.35)


def slow_move_to_joints(q, target, grip=None, seconds=2.5, steps=150):
    cj, cg = q.read_all()
    cj = np.array(cj, dtype=float)
    cg = float(cg)
    g_target = cg if grip is None else float(grip)
    for i in range(1, steps + 1):
        a = i / steps
        s = 3 * a * a - 2 * a * a * a
        q.set_joints_and_gripper(
            cj + s * (target - cj),
            cg + s * (g_target - cg),
        )
        time.sleep(seconds / steps)
    for _ in range(15):
        q.set_joints_and_gripper(target, g_target)
        time.sleep(0.05)


def ik_safe(xyz, gamma=0.0):
    try:
        phi = inverse_kinematics(xyz, gamma=gamma)
    except Exception as ex:
        return None, f"IK error: {ex}"
    phi = np.array(phi, dtype=float)
    for i in range(4):
        lo, hi = JOINT_LIMITS[i]
        if phi[i] < lo or phi[i] > hi:
            return None, f"joint {i} out of limits ({np.degrees(phi[i]):+.1f} deg)"
    return phi, None


def detect_strawberries(cam, T, min_area=fd.MIN_AREA):
    """Capture one stable frame, detect strawberries, project to base."""
    # a few frames so exposure is fresh
    for _ in range(5):
        try: cam.read()
        except Exception: pass
        time.sleep(0.04)
    color, depth = cam.read()
    dets = fd.detect_fruits(color, depth, min_area=min_area)
    out = []
    for d in dets:
        if d.fruit_type not in ("strawberry", "tomato"):
            continue
        row, col = d.centroid
        d_mm = detection_depth_mm(d, depth)
        if d_mm is None:
            continue
        xyz = cam.pixel_to_world(row, col, d_mm, T)
        out.append({
            "type": d.fruit_type,
            "pix": (col, row),
            "xyz": xyz,
            "area": d.area,
        })
    return color, depth, out


def main():
    with open(POINTS_FILE) as f:
        pts = json.load(f)
    with open(CALIB_FILE) as f:
        cal = json.load(f)
    T = np.array(cal["T_cam_to_base"], dtype=float)
    print(f"Calibration: {cal.get('source', 'unknown')}  "
          f"RMS={cal.get('rms_residual_mm', '?')} mm")

    q = QArmDriver()
    q.connect()
    time.sleep(0.5)
    for _ in range(5):
        try: q.read_all(); break
        except Exception: time.sleep(0.5)

    try:
        # Move to pickhome1 for detection view
        if HOME_LABEL in pts:
            home_joints = np.array(pts[HOME_LABEL]["joints_rad"], dtype=float)
            home_grip = float(pts[HOME_LABEL].get("gripper", 0.15))
            print(f"\n-> moving to {HOME_LABEL} for detection...")
            slow_move_to_joints(q, home_joints, home_grip, seconds=3.0)

        cam = QArmCamera()
        cam.open()
        try:
            # warmup
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    c, d = cam.read()
                    if c.mean() > 5 and (d > 0).mean() > 0.1:
                        break
                except Exception: pass
                time.sleep(0.1)

            color, depth, dets = detect_strawberries(cam, T)
            os.makedirs(LOG_DIR, exist_ok=True)
            out_img = os.path.join(LOG_DIR, "hover_test_scene.png")
            cv2.imwrite(out_img, fd.draw_detections(color,
                        fd.detect_fruits(color, depth)))
            print(f"scene -> {out_img}")
            print(f"detected {len(dets)} red fruits")
            for i, d in enumerate(dets):
                print(f"  [{i}] {d['type']:10s} pix={d['pix']}  "
                      f"xyz={d['xyz'].round(3)}")

            # filter to workspace
            targets = []
            for d in dets:
                x, y, z = d["xyz"]
                if not (WS_X[0] <= x <= WS_X[1]) or \
                   not (WS_Y[0] <= y <= WS_Y[1]):
                    print(f"  [skip] {d['type']} at {np.round(d['xyz'],3)}"
                          " outside workspace")
                    continue
                targets.append(d)

            if not targets:
                print("no valid targets, returning home")
                return

            # sort by x descending (far first), then y (-Y first)
            targets.sort(key=lambda d: (-d["xyz"][0], d["xyz"][1]))

            print(f"\nwill hover over {len(targets)} targets in order:")
            for i, d in enumerate(targets):
                print(f"  {i+1}. {d['type']} at xyz={d['xyz'].round(3)}")

            # Current pose (pickhome1)
            current_joints, _ = q.read_all()
            current_joints = np.array(current_joints, dtype=float)

            for i, d in enumerate(targets):
                tx, ty, tz = d["xyz"]
                hover_z = max(tz + HOVER_DZ, SAFE_MIN_Z)
                transit_z = max(SAFE_TRANSIT_Z, hover_z + 0.05)

                print(f"\n--- Target {i+1}/{len(targets)}: "
                      f"xyz=({tx:+.3f},{ty:+.3f},{tz:+.3f}) ---")

                # Waypoint above current xy
                cur_xyz, _ = forward_kinematics(current_joints)
                # 1) ascend to transit z at current xy
                up1_xyz = np.array([cur_xyz[0], cur_xyz[1], transit_z])
                phi_up1, err = ik_safe(up1_xyz)
                if err:
                    print(f"  [skip] IK fail ascending: {err}")
                    continue
                print(f"  ascend to transit z ...")
                slow_move_to_joints(q, phi_up1, seconds=1.5)
                current_joints = phi_up1.copy()

                # 2) lateral move to target xy at transit z
                wp_xyz = np.array([tx, ty, transit_z])
                phi_wp, err = ik_safe(wp_xyz)
                if err:
                    print(f"  [skip] IK fail at waypoint: {err}")
                    continue
                print(f"  lateral to (x={tx:+.3f}, y={ty:+.3f}, "
                      f"z={transit_z:.3f}) ...")
                slow_move_to_joints(q, phi_wp, seconds=2.5)
                current_joints = phi_wp.copy()

                # 3) descend to hover z
                hv_xyz = np.array([tx, ty, hover_z])
                phi_hv, err = ik_safe(hv_xyz)
                if err:
                    print(f"  [skip] IK fail at hover: {err}")
                    continue
                print(f"  descend to hover z={hover_z:.3f} ...")
                slow_move_to_joints(q, phi_hv, seconds=2.0)
                current_joints = phi_hv.copy()

                # report FK tip
                tip_xyz, _ = forward_kinematics(current_joints)
                tip_xyz = np.array(tip_xyz, dtype=float)
                dxy = np.hypot(tip_xyz[0] - tx, tip_xyz[1] - ty) * 1000
                print(f"  [hover] tip FK xyz={tip_xyz.round(3)}  "
                      f"dxy vs target = {dxy:.1f} mm")
                print(f"  dwelling {DWELL_SEC:.1f}s for visual check...")
                time.sleep(DWELL_SEC)

                # 4) ascend back to transit z
                phi_up2, err = ik_safe(wp_xyz)
                if err is None:
                    slow_move_to_joints(q, phi_up2, seconds=1.5)
                    current_joints = phi_up2.copy()

            # Return to pickhome1
            print("\n-> returning to pickhome1 ...")
            slow_move_to_joints(q, home_joints, home_grip, seconds=3.0)
            print("done.")

        finally:
            cam.close()
    finally:
        try:
            q.card.close(); q._connected = False
        except Exception: pass
        print("QArm released (held at last pose).")


if __name__ == "__main__":
    main()
