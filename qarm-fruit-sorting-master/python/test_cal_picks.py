"""Pick-and-release test using cal_01..cal_04 teach points directly.

Bypasses camera detection and IK — uses joint angles from teach_points.json,
so this isolates the pick *mechanics* (gripper timing, motion, lift) from
any calibration error. UGreen frames are captured at every stage for visual
verification.

Flow per fruit:
  1. UGreen snapshot at pickhome1 (arm clear of target area)
  2. Descend to cal_X (TCP at fruit) with gripper OPEN
  3. Close gripper (ramped)
  4. Read gripper readback (< 1.0 suggests holding something)
  5. Lift back to pickhome1 (UGreen can now see if fruit disappeared)
  6. UGreen snapshot — if cal_X area is empty, fruit is in gripper
  7. Descend back to cal_X (release at same spot)
  8. Open gripper — fruit drops back
  9. Lift to pickhome1 — continue

Outputs:
  logs/cal_pick_test/00_initial.png
  logs/cal_pick_test/NN_cal_0N_{before,lifted,released}.png
  logs/cal_pick_test/99_final.png
  Terminal report with per-fruit gripper readback.
"""
import json
import os
import time
import numpy as np
import cv2

from qarm_driver import QArmDriver
from qarm_kinematics import forward_kinematics, inverse_kinematics, JOINT_LIMITS
from ugreen_tracker import capture

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS_FILE = os.path.join(REPO, "teach_points.json")
OUT_DIR = os.path.join(REPO, "logs", "cal_pick_test")

LABELS = ["cal01", "cal02", "cal03", "cal04"]
PICKHOME = "pickhome1"
GRIP_CLOSE = 0.9
GRIP_OPEN = 0.0


def slow_move(q, target_joints, target_grip, seconds=3.0, steps=200):
    cj, cg = q.read_all()
    cj = np.array(cj, dtype=float)
    cg = float(cg)
    for i in range(1, steps + 1):
        a = i / steps
        s = 3 * a * a - 2 * a * a * a
        q.set_joints_and_gripper(
            cj + s * (target_joints - cj),
            cg + s * (target_grip - cg),
        )
        time.sleep(seconds / steps)
    for _ in range(15):
        q.set_joints_and_gripper(target_joints, target_grip)
        time.sleep(0.05)


def ramp_gripper(q, joints, start, end, steps=25, dt=0.04):
    for a in np.linspace(start, end, steps):
        q.set_joints_and_gripper(joints, float(a))
        time.sleep(dt)
    for _ in range(10):
        q.set_joints_and_gripper(joints, end)
        time.sleep(0.05)


def ik_safe(xyz, gamma=0.0):
    """IK with joint-limit check. Returns (phi, None) or (None, err_msg)."""
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


def approach_staged(q, cal_joints, cal_xyz, grip, transit_dz=0.15, seconds=2.5):
    """Move to cal_xyz in 2 Cartesian stages (NOT joint-space interp):
      1) lateral: rotate+advance at current Z to (cal_x, cal_y, current_z)
      2) descend: straight down to (cal_x, cal_y, cal_z)

    Keeps Z high until final descend. Falls back to direct joint-space
    move if IK fails for the waypoint.
    """
    cj, _ = q.read_all()
    cj = np.array(cj, dtype=float)
    cur_xyz, _ = forward_kinematics(cj)
    cur_xyz = np.array(cur_xyz, dtype=float)

    transit_z = max(cur_xyz[2], cal_xyz[2] + transit_dz)

    wp_xyz = np.array([cal_xyz[0], cal_xyz[1], transit_z])
    phi_wp, err = ik_safe(wp_xyz, gamma=0.0)
    if err:
        print(f"    [warn] IK failed for waypoint {wp_xyz.round(3)}: {err}")
        print(f"    falling back to direct joint-space move")
        slow_move(q, cal_joints, grip, seconds=seconds * 1.5)
        return

    print(f"    stage 1: rotate+advance to waypoint xyz={wp_xyz.round(3)}")
    slow_move(q, phi_wp, grip, seconds=seconds)
    print(f"    stage 2: descend straight down to cal xyz={cal_xyz.round(3)}")
    slow_move(q, cal_joints, grip, seconds=seconds * 0.8)


def retreat_staged(q, cal_joints, cal_xyz, ph_joints, grip,
                    transit_dz=0.15, seconds=2.5):
    """Reverse of approach_staged: ascend straight up first, then lateral
    to pickhome."""
    transit_xyz = np.array([cal_xyz[0], cal_xyz[1], cal_xyz[2] + transit_dz])
    phi_up, err = ik_safe(transit_xyz, gamma=0.0)
    if err:
        print(f"    [warn] IK failed for retreat waypoint: {err}")
        print(f"    falling back to direct joint-space move to pickhome1")
        slow_move(q, ph_joints, grip, seconds=seconds * 1.5)
        return
    print(f"    stage 1: ascend straight up to xyz={transit_xyz.round(3)}")
    slow_move(q, phi_up, grip, seconds=seconds * 0.8)
    print(f"    stage 2: lateral to pickhome1")
    slow_move(q, ph_joints, grip, seconds=seconds)


def snap(tag):
    """Capture UGreen, save to OUT_DIR with given tag, return path."""
    frame = capture()
    path = os.path.join(OUT_DIR, f"{tag}.png")
    cv2.imwrite(path, frame)
    return path, frame


def main():
    with open(POINTS_FILE) as f:
        pts = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    for lbl in [PICKHOME] + LABELS:
        if lbl not in pts:
            print(f"ERROR: teach point {lbl} missing")
            return

    q = QArmDriver()
    q.connect()
    time.sleep(0.5)
    for _ in range(5):
        try:
            q.read_all(); break
        except Exception:
            time.sleep(0.3)

    ph_joints = np.array(pts[PICKHOME]["joints_rad"], dtype=float)

    results = []

    try:
        print(f"Moving to {PICKHOME} (starting pose)...")
        slow_move(q, ph_joints, GRIP_OPEN, seconds=3.0)

        snap("00_initial")
        print(f"Initial scene captured -> {OUT_DIR}/00_initial.png")

        for i, lbl in enumerate(LABELS, 1):
            cal_joints = np.array(pts[lbl]["joints_rad"], dtype=float)
            cal_xyz = np.array(pts[lbl]["xyz_m"], dtype=float)
            print(f"\n--- {i}/4: {lbl} @ xyz={cal_xyz.round(3)} ---")

            snap(f"{i:02d}_{lbl}_01_before")

            print(f"  [1] approach {lbl} (rotate+advance, then descend)...")
            approach_staged(q, cal_joints, cal_xyz, GRIP_OPEN)
            time.sleep(1.0)

            snap(f"{i:02d}_{lbl}_02_at_position")
            print(f"      at-position frame -> "
                  f"{OUT_DIR}/{i:02d}_{lbl}_02_at_position.png")

            # NOTE: gripper steps intentionally skipped — this run only
            # validates the staged approach trajectory and UGreen feedback.
            # Re-enable ramp_gripper() calls once cal_XX pick positions are
            # finalised.
            _, gr = q.read_all()
            print(f"      gripper readback (unchanged): {gr:.3f}")

            print(f"  [2] retreat to {PICKHOME} (ascend, then lateral)...")
            retreat_staged(q, cal_joints, cal_xyz, ph_joints, GRIP_OPEN)
            time.sleep(0.5)

            snap(f"{i:02d}_{lbl}_03_retreated")
            print(f"      retreated frame -> "
                  f"{OUT_DIR}/{i:02d}_{lbl}_03_retreated.png")

            results.append({
                "label": lbl,
                "xyz": cal_xyz.tolist(),
                "grip_readback": float(gr),
            })

        snap("99_final")
        print(f"\nFinal scene -> {OUT_DIR}/99_final.png")

        print("\n" + "=" * 60)
        print("  RESULTS (approach-only, no gripper)")
        print("=" * 60)
        for r in results:
            print(f"  {r['label']}  xyz={np.round(r['xyz'],3)}  "
                  f"grip_readback={r['grip_readback']:.3f}")
        print(f"\nAll frames in: {OUT_DIR}")
        print("Check NN_cal_XX_02_at_position.png to verify TCP landed over fruit")
        print("Check NN_cal_XX_03_retreated.png to verify nothing got disturbed")

    finally:
        try:
            q.card.close(); q._connected = False
        except Exception:
            pass
        print("\nQArm released (held at last pose).")


if __name__ == "__main__":
    main()
