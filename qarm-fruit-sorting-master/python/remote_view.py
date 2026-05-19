"""OpenCV companion window for remote mode. Launched by Simulink when
remote mode activates; closed via cv2.setMouseCallback click or ESC.

Reads live D415 frames and overlays the current mode, current joints,
current xyz, gripper state, jog step, and the last HMI command. The
current state is supplied by Simulink through a shared JSON status
file (polled at 10 Hz) to avoid in-process Python<->MATLAB coupling.
"""
import json
import os
import sys
import time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
STATUS_FILE = os.path.join(REPO, "logs", "remote_status.json")
WINDOW = "QArm Remote - D415 live"


def read_status():
    if not os.path.exists(STATUS_FILE):
        return None
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    from camera import QArmCamera
    cam = QArmCamera()
    cam.open()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            try:
                color, _ = cam.read()
            except Exception:
                color = np.zeros((720, 1280, 3), dtype=np.uint8)
            disp = color.copy()
            s = read_status()
            if s is None:
                cv2.putText(disp, "waiting for Simulink status...",
                             (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                             (0, 255, 255), 2)
            else:
                lines = [
                    f"mode     : {s.get('mode','?')}",
                    f"sub-mode : {s.get('sub_mode','?')}",
                    f"joints(d): {s.get('joints_deg','?')}",
                    f"xyz (m)  : {s.get('xyz_m','?')}",
                    f"gripper  : {s.get('gripper','?')}",
                    f"step     : {s.get('step','?')}",
                    f"last cmd : {s.get('last_cmd','?')}",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(disp, line, (20, 30 + i * 26),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                 (0, 0, 0), 4)
                    cv2.putText(disp, line, (20, 30 + i * 26),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                 (0, 255, 255), 2)
            cv2.imshow(WINDOW, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:      # ESC
                break
    finally:
        try: cam.close()
        except Exception: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
