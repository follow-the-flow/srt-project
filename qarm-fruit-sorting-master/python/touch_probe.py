"""
Interactive jog-and-confirm helper for the session calibration.

Two public entry points:

  capture_tcp(driver) - non-interactive: reads current joints, returns
                        FK(joints)[0] = TCP position in base frame.
                        Used in tests + any flow where the operator has
                        already positioned the arm manually.

  jog_and_capture(driver) - the interactive flow used by
                            calibrate_chessboard.py: runs a cv2.waitKeyEx
                            loop so the operator can jog the gripper to
                            the chessboard origin corner, then returns
                            the TCP on ENTER. Imports cv2 lazily so the
                            unit tests don't drag in a UI dependency.
"""
from __future__ import annotations
import numpy as np

from qarm_kinematics import forward_kinematics


def capture_tcp(driver) -> np.ndarray:
    """Return TCP position in base frame from the driver's current joints.
    Shape (3,) float64, units: metres."""
    joints, _ = driver.read_all()
    pos, _ = forward_kinematics(np.asarray(joints, dtype=float))
    return np.asarray(pos, dtype=float).flatten()[:3]


def jog_and_capture(driver, jog_step_rad: float = 0.02,
                     jog_step_grip: float = 0.05) -> np.ndarray:
    """
    Open a minimal HUD window and run a jog loop using cv2.waitKeyEx.
    Keys q/w/e/r step joints 1-4 positively; a/s/d/f step them negatively.
    z/x close/open the gripper. ENTER returns the current TCP. ESC raises
    KeyboardInterrupt.

    Uses the same cv2.waitKeyEx paradigm as the existing teach_points.py
    - cross-platform (no termios/msvcrt), works wherever OpenCV does.
    """
    import cv2  # lazy so unit tests don't require it.

    joints, grip = driver.read_all()
    joints = np.asarray(joints, dtype=float).copy()
    grip = float(grip)

    step_map = {
        ord("q"): (0, +1), ord("a"): (0, -1),
        ord("w"): (1, +1), ord("s"): (1, -1),
        ord("e"): (2, +1), ord("d"): (2, -1),
        ord("r"): (3, +1), ord("f"): (3, -1),
    }
    win = "Touch probe - jog to chessboard origin"
    cv2.namedWindow(win)
    try:
        while True:
            hud = np.zeros((260, 680, 3), dtype=np.uint8)
            lines = [
                f"joints (rad): {joints[0]:+.3f} {joints[1]:+.3f} "
                f"{joints[2]:+.3f} {joints[3]:+.3f}",
                f"gripper:      {grip:.2f}",
                "q/w/e/r = +j1/j2/j3/j4,  a/s/d/f = - ,  z = close, x = open",
                "ENTER = confirm origin corner,  ESC = abort",
            ]
            for i, line in enumerate(lines):
                cv2.putText(hud, line, (10, 40 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA)
            cv2.imshow(win, hud)
            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            if key in (13, 10):                   # ENTER
                return capture_tcp(driver)
            if key == 27:                          # ESC
                raise KeyboardInterrupt("jog aborted by operator")
            if key == ord("z"):
                grip = min(0.90, grip + jog_step_grip)  # z = close
            elif key == ord("x"):
                grip = max(0.15, grip - jog_step_grip)  # x = open
            elif key in step_map:
                idx, sign = step_map[key]
                joints[idx] += sign * jog_step_rad
            else:
                continue
            driver.set_joints_and_gripper(joints, grip)
    finally:
        cv2.destroyWindow(win)
