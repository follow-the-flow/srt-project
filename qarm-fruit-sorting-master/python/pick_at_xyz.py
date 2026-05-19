"""
Manual-coordinate pick: you supply XYZ in base frame directly, script
does the HOVER -> DESCEND -> CLOSE -> LIFT sequence. Bypasses the
vision pipeline entirely.

Usage:
    py -3.13 python/pick_at_xyz.py --xyz 0.5,0.1,0.05

XYZ = fruit TOP centre in base frame, metres. Typical tomato: z≈0.05.

To measure XYZ: run touch_probe-style jog first, place gripper tip at
top centre of fruit, read TCP. Or physically measure with a ruler from
a known reference (chessboard corner or base).
"""
import argparse, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qarm_kinematics import inverse_kinematics, JOINT_LIMITS
from sorting_controller import GRIP_OPEN, GRIP_CLOSE

APPROACH_Z = 0.20
DESCEND_OFFSET = 0.02
PICK_Z_FLOOR = 0.03


def _solve_or_abort(label, xyz):
    try:
        joints = inverse_kinematics(xyz, gamma=0.0)
    except Exception as ex:
        sys.exit(f"[abort] IK for {label} @ {xyz}: {ex}")
    for i, j in enumerate(joints):
        lo, hi = JOINT_LIMITS[i]
        if not (lo - 0.01 <= j <= hi + 0.01):
            sys.exit(f"[abort] {label} joint {i}={j:.3f} outside [{lo:.3f}, {hi:.3f}]")
    return np.asarray(joints, dtype=float)


def _wait(prompt):
    try:
        input(f"  {prompt} (ENTER to go / Ctrl+C to abort)... ")
    except KeyboardInterrupt:
        sys.exit("[abort] user cancelled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True,
                     help="target XYZ in base frame, e.g. 0.5,0.1,0.05")
    args = ap.parse_args()
    xyz = np.array([float(v) for v in args.xyz.split(",")], dtype=float)
    if xyz.shape != (3,):
        sys.exit("--xyz must be X,Y,Z")

    fx, fy, fz = xyz.tolist()
    pick_z = max(PICK_Z_FLOOR, fz - DESCEND_OFFSET)
    print(f"  target top  = ({fx:+.3f}, {fy:+.3f}, {fz:+.3f})")
    print(f"  pick_z      = {pick_z:.3f}  (floor {PICK_Z_FLOOR}, "
          f"descend offset {DESCEND_OFFSET})")

    hover_xyz = np.array([fx, fy, APPROACH_Z])
    pick_xyz = np.array([fx, fy, pick_z])
    lift_xyz = np.array([fx, fy, APPROACH_Z])

    hover_j = _solve_or_abort("HOVER", hover_xyz)
    pick_j = _solve_or_abort("PICK", pick_xyz)
    lift_j = _solve_or_abort("LIFT", lift_xyz)
    print("  IK OK for all 3 poses.\n")

    from qarm_driver import QArmDriver
    from calibrate_closed_loop import slow_move_to_joints

    driver = QArmDriver()
    driver.connect(); time.sleep(0.3)
    try:
        print(f"  [1] HOVER @ z={APPROACH_Z:.3f}")
        _wait("HOVER")
        slow_move_to_joints(driver, hover_j, GRIP_OPEN)

        print(f"\n  [2] DESCEND @ z={pick_z:.3f}")
        _wait("DESCEND")
        slow_move_to_joints(driver, pick_j, GRIP_OPEN)

        print(f"\n  [3] CLOSE gripper -> {GRIP_CLOSE}")
        _wait("CLOSE GRIPPER")
        joints_now, _ = driver.read_all()
        for i in range(31):
            s = i / 30.0
            g = GRIP_OPEN + s * (GRIP_CLOSE - GRIP_OPEN)
            driver.set_joints_and_gripper(joints_now, g)
            time.sleep(0.03)
        time.sleep(0.5)
        _, g_act = driver.read_all()
        print(f"    readback {g_act:.3f}  (commanded {GRIP_CLOSE})")

        print(f"\n  [4] LIFT @ z={APPROACH_Z:.3f}")
        _wait("LIFT")
        slow_move_to_joints(driver, lift_j, GRIP_CLOSE)

        print("\n  DONE. Gripper holds at lift pose with fruit (hopefully).")
        print("  To release: drive gripper to 0.15 then return home.")
    finally:
        try: driver.card.close()
        except Exception: pass
        driver._connected = False


if __name__ == "__main__":
    main()
