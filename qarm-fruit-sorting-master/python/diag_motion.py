"""
Minimal motion diagnostic:
  - Connect, read actual joints.
  - Command a tiny +2 deg yaw change (j0), then back.
  - After each command, read actual joints and report commanded vs actual.
If actual doesn't follow commanded, motors are in fault or disabled.
"""

import time
import numpy as np
from qarm_driver import QArmDriver


def main():
    q = QArmDriver()
    q.connect()
    time.sleep(0.5)
    for _ in range(5):
        try:
            q.read_all(); break
        except Exception:
            time.sleep(0.5)
    try:
        cur, grip = q.read_all()
        cur = np.array(cur, dtype=float)
        print(f"[initial] actual joints (deg): {np.degrees(cur).round(2)}")
        print(f"[initial] gripper: {grip:.3f}")

        # Nudge yaw by +2 deg, over 2 s (slow)
        target = cur.copy()
        target[0] += np.deg2rad(2.0)
        steps = 50
        for i in range(1, steps + 1):
            a = i / steps
            phi = cur + a * (target - cur)
            q.set_joints_and_gripper(phi, grip)
            time.sleep(0.04)

        # Let it settle
        for _ in range(10):
            q.set_joints_and_gripper(target, grip)
            time.sleep(0.05)

        act, g_act = q.read_all()
        act = np.array(act, dtype=float)
        err_deg = np.degrees(act - target)
        print(f"\n[after +2deg yaw]")
        print(f"  commanded joints (deg): {np.degrees(target).round(2)}")
        print(f"  actual    joints (deg): {np.degrees(act).round(2)}")
        print(f"  error     joints (deg): {err_deg.round(3)}")

        # Bring it back
        for i in range(1, steps + 1):
            a = i / steps
            phi = target + a * (cur - target)
            q.set_joints_and_gripper(phi, grip)
            time.sleep(0.04)
        for _ in range(10):
            q.set_joints_and_gripper(cur, grip)
            time.sleep(0.05)

        back, _ = q.read_all()
        back = np.array(back, dtype=float)
        err_back = np.degrees(back - cur)
        print(f"\n[after return]")
        print(f"  commanded joints (deg): {np.degrees(cur).round(2)}")
        print(f"  actual    joints (deg): {np.degrees(back).round(2)}")
        print(f"  error     joints (deg): {err_back.round(3)}")

        # Verdict
        worst = max(abs(err_deg).max(), abs(err_back).max())
        print(f"\nworst |cmd - actual| = {worst:.3f} deg")
        if worst > 0.5:
            print("-> Motors NOT tracking commands. "
                  "QArm likely in fault / motors disabled.")
        else:
            print("-> Motors tracking fine (< 0.5 deg error).")

    finally:
        try:
            q.card.close(); q._connected = False
        except Exception:
            pass


if __name__ == "__main__":
    main()
