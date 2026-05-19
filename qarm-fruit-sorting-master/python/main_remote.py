#!/usr/bin/env python3
"""
Main script: Remote Control (Teleoperation) Mode for QArm.

Controls the QArm via keyboard input:
    Arrow keys: Move end-effector in X/Y
    Q/A: Move end-effector up/down (Z)
    Space: Toggle gripper
    H: Return to home
    ESC: Exit

Usage:
    python main_remote.py

Requires:
    - QArm connected via USB
    - Quanser SDK Python API installed
"""

import sys
import numpy as np
import time

from qarm_driver import QArmDriver
from qarm_kinematics import forward_kinematics, inverse_kinematics


# Speed settings
EE_SPEED = 0.05  # m/s per key press
DT = 0.002       # Control loop period (500 Hz)


def main():
    print("=" * 60)
    print("  QArm - Remote Control (Teleoperation)")
    print("  Quanser SDK for Windows")
    print("=" * 60)
    print()
    print("  Controls:")
    print("    Up/Down Arrow  : Move X (forward/back)")
    print("    Left/Right Arrow: Move Y (left/right)")
    print("    Q / A          : Move Z (up/down)")
    print("    Space          : Toggle gripper")
    print("    H              : Return to home")
    print("    ESC            : Exit")
    print()

    # --- Connect to QArm ---
    qarm = QArmDriver()
    try:
        qarm.connect()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    try:
        qarm.home(duration=3.0)
        time.sleep(0.5)

        gripper_closed = False

        # Try to use Quanser Keyboard device
        try:
            from quanser.devices import Keyboard, VirtualKeyCodes
            kb = Keyboard()
            use_quanser_kb = True
            print("Using Quanser Keyboard device")
        except Exception:
            use_quanser_kb = False
            print("Quanser Keyboard not available.")
            print("Using simple input loop instead.")
            print("Type commands: w/s/a/d/q/e/g/h or 'exit'")

        print("\nRemote control active!\n")

        if use_quanser_kb:
            # --- Real-time keyboard control loop ---
            while True:
                loop_start = time.time()

                # Read current state
                joints, grip = qarm.read_all()
                ee_pos, _ = forward_kinematics(joints)

                # Check keys
                vx, vy, vz = 0.0, 0.0, 0.0

                buf = kb.createBuffer(10)
                keys = [
                    VirtualKeyCodes.UP, VirtualKeyCodes.DOWN,
                    VirtualKeyCodes.LEFT, VirtualKeyCodes.RIGHT,
                    VirtualKeyCodes.Q, VirtualKeyCodes.A,
                    VirtualKeyCodes.SPACE, VirtualKeyCodes.H,
                    VirtualKeyCodes.ESCAPE
                ]
                states = kb.getKeyStates(keys, buf)

                if states[8]:  # ESC
                    break
                if states[0]:  # UP
                    vx -= EE_SPEED
                if states[1]:  # DOWN
                    vx += EE_SPEED
                if states[2]:  # LEFT
                    vy -= EE_SPEED
                if states[3]:  # RIGHT
                    vy += EE_SPEED
                if states[4]:  # Q
                    vz += EE_SPEED
                if states[5]:  # A
                    vz -= EE_SPEED
                if states[6]:  # SPACE (toggle)
                    gripper_closed = not gripper_closed
                    time.sleep(0.2)  # Debounce
                if states[7]:  # H (home)
                    qarm.home(duration=2.0)
                    continue

                # Compute new position
                dp = np.array([vx, vy, vz]) * DT
                target_pos = ee_pos + dp

                # IK and execute
                try:
                    phi = inverse_kinematics(target_pos, gamma=0.0)
                    qarm.set_joints_and_gripper(phi, 1.0 if gripper_closed else 0.0)
                except Exception:
                    pass  # Skip if IK fails (out of workspace)

                # Status display (every 500ms)
                if int(time.time() * 2) % 2 == 0:
                    sys.stdout.write(
                        f"\r  EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] "
                        f"  Gripper: {'CLOSED' if gripper_closed else 'OPEN'}   "
                    )
                    sys.stdout.flush()

                # Maintain loop rate
                elapsed = time.time() - loop_start
                if DT - elapsed > 0:
                    time.sleep(DT - elapsed)

        else:
            # --- Simple text input fallback ---
            while True:
                joints, grip = qarm.read_all()
                ee_pos, _ = forward_kinematics(joints)
                print(f"  EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] "
                      f"Gripper: {'CLOSED' if gripper_closed else 'OPEN'}")

                cmd = input("  Command (w/s/a/d/q/e/g/h/exit): ").strip().lower()
                step = 0.02  # 2cm per command

                if cmd == 'exit':
                    break
                elif cmd == 'w': ee_pos[0] -= step
                elif cmd == 's': ee_pos[0] += step
                elif cmd == 'a': ee_pos[1] -= step
                elif cmd == 'd': ee_pos[1] += step
                elif cmd == 'q': ee_pos[2] += step
                elif cmd == 'e': ee_pos[2] -= step
                elif cmd == 'g':
                    gripper_closed = not gripper_closed
                elif cmd == 'h':
                    qarm.home(duration=2.0)
                    continue
                else:
                    print("    Unknown command")
                    continue

                try:
                    phi = inverse_kinematics(ee_pos, gamma=0.0)
                    qarm.set_joints_and_gripper(phi, 1.0 if gripper_closed else 0.0)
                    time.sleep(0.5)
                except Exception as ex:
                    print(f"    IK failed: {ex}")

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    finally:
        print("\nReturning home...")
        qarm.home(duration=3.0)
        qarm.disconnect()


if __name__ == "__main__":
    main()
