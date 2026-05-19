"""Per-joint jog tool for QArm. Move each of the 4 joints independently
with the keyboard. Shows live FK position for reference.

Keys (focus the OpenCV window):
  q / a  :  J0 base       -step / +step
  w / s  :  J1 shoulder   -step / +step
  e / d  :  J2 elbow      -step / +step
  r / f  :  J3 wrist      -step / +step
  SPACE  :  toggle gripper (open <-> closed)
  [ / ]  :  halve / double step size
  h      :  slow home (5 s)
  p      :  print current joints + xyz to terminal
  ESC    :  exit (does not save anything)

Joints are clamped to JOINT_LIMITS automatically. Step defaults to 3 deg.
"""
import time
import numpy as np
import cv2

from qarm_driver import QArmDriver
from qarm_kinematics import forward_kinematics, JOINT_LIMITS

WINDOW = "QArm Joint Jog - focus here to drive (ESC to exit)"

DEFAULT_STEP_DEG = 3.0
STEP_MIN_DEG = 0.5
STEP_MAX_DEG = 15.0

GRIP_OPEN = 0.2
GRIP_CLOSE = 0.9

JOINT_NAMES = ["J0 base", "J1 shoulder", "J2 elbow", "J3 wrist"]


def clamp(val, limits):
    lo, hi = limits
    return max(lo, min(hi, val))


def render(joints, grip, step_rad, status):
    disp = np.zeros((420, 780, 3), dtype=np.uint8)
    xyz, _ = forward_kinematics(joints)
    xyz = np.array(xyz, dtype=float)

    def put(text, y, color=(255, 255, 255), scale=0.6):
        cv2.putText(disp, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[i]
        v = np.degrees(joints[i])
        near_lo = abs(joints[i] - lo) < np.deg2rad(3)
        near_hi = abs(joints[i] - hi) < np.deg2rad(3)
        color = (0, 0, 255) if (near_lo or near_hi) else (255, 255, 255)
        put(f"{name:12s}: {v:+8.2f} deg   "
            f"[{np.degrees(lo):+6.1f}, {np.degrees(hi):+6.1f}]",
            30 + i * 26, color)

    put(f"TCP xyz       : [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}] m",
        155, (0, 255, 255))
    put(f"gripper       : {grip:.3f}  "
        f"({'CLOSED' if grip > 0.5 else 'OPEN'})",
        185, (0, 255, 255))
    put(f"step size     : {np.degrees(step_rad):.1f} deg", 215)

    put("Keys:", 265, (180, 180, 180))
    put("q/a J0   w/s J1   e/d J2   r/f J3", 290)
    put("SPACE grip   [ / ] step   h home   p print   ESC exit", 315)

    if status:
        put(status, 370, (120, 220, 120))
    return disp


def main():
    q = QArmDriver()
    q.connect()
    time.sleep(0.3)
    for _ in range(5):
        try:
            q.read_all(); break
        except Exception:
            time.sleep(0.3)

    joints_init, grip_init = q.read_all()
    joints = np.array(joints_init, dtype=float)
    grip = float(grip_init)
    step = np.deg2rad(DEFAULT_STEP_DEG)
    status = "connected - start jogging"

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            cv2.imshow(WINDOW, render(joints, grip, step, status))
            k = cv2.waitKey(30) & 0xFF

            # Hold current pose every frame even if no key pressed
            q.set_joints_and_gripper(joints, grip)

            if k == 255:
                continue  # no key

            joint_idx = None
            delta = 0.0

            if k == ord('q'):
                joint_idx, delta = 0, -step
            elif k == ord('a'):
                joint_idx, delta = 0, +step
            elif k == ord('w'):
                joint_idx, delta = 1, -step
            elif k == ord('s'):
                joint_idx, delta = 1, +step
            elif k == ord('e'):
                joint_idx, delta = 2, -step
            elif k == ord('d'):
                joint_idx, delta = 2, +step
            elif k == ord('r'):
                joint_idx, delta = 3, -step
            elif k == ord('f'):
                joint_idx, delta = 3, +step
            elif k == 32:  # SPACE
                grip = GRIP_OPEN if grip > 0.5 else GRIP_CLOSE
                status = f"gripper -> {'OPEN' if grip < 0.5 else 'CLOSED'}"
            elif k == ord('['):
                step = max(np.deg2rad(STEP_MIN_DEG), step / 2)
                status = f"step = {np.degrees(step):.1f} deg"
            elif k == ord(']'):
                step = min(np.deg2rad(STEP_MAX_DEG), step * 2)
                status = f"step = {np.degrees(step):.1f} deg"
            elif k == ord('h'):
                status = "homing (5 s)..."
                cv2.imshow(WINDOW, render(joints, grip, step, status))
                cv2.waitKey(1)
                q.home(duration=5.0)
                j2, g2 = q.read_all()
                joints = np.array(j2, dtype=float)
                grip = float(g2)
                status = "homed"
            elif k == ord('p'):
                xyz, _ = forward_kinematics(joints)
                print(f"joints_rad = {joints.tolist()}")
                print(f"joints_deg = {np.degrees(joints).tolist()}")
                print(f"xyz_m      = {list(xyz)}")
                print(f"gripper    = {grip}")
                status = "printed to terminal"
            elif k == 27:  # ESC
                break

            if joint_idx is not None:
                new_val = clamp(joints[joint_idx] + delta,
                                JOINT_LIMITS[joint_idx])
                if abs(new_val - (joints[joint_idx] + delta)) > 1e-9:
                    status = (f"{JOINT_NAMES[joint_idx]} clamped at limit "
                              f"({np.degrees(new_val):+.1f} deg)")
                else:
                    status = (f"{JOINT_NAMES[joint_idx]} -> "
                              f"{np.degrees(new_val):+.1f} deg")
                joints[joint_idx] = new_val

    finally:
        cv2.destroyAllWindows()
        try:
            q.card.close(); q._connected = False
        except Exception:
            pass
        print("QArm released (held at last pose).")


if __name__ == "__main__":
    main()
