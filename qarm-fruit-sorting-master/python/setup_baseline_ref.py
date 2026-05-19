"""One-shot helper: move arm to homeplace0, capture UGreen baseline,
move to pickhome1, capture reference frame + TCP pixel.

Writes:
  logs/ugreen_baseline.png
  logs/ugreen_pickhome1_reference.png
  logs/ugreen_pickhome1_tcp.json
"""
import json
import os
import sys
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
POINTS_FILE = os.path.join(REPO, "teach_points.json")
LOGS = os.path.join(REPO, "logs")
BASELINE_PNG = os.path.join(LOGS, "ugreen_baseline.png")
REF_PNG = os.path.join(LOGS, "ugreen_pickhome1_reference.png")
REF_TCP = os.path.join(LOGS, "ugreen_pickhome1_tcp.json")


def main():
    os.makedirs(LOGS, exist_ok=True)
    from qarm_driver import QArmDriver
    from ugreen_tracker import capture, tcp_from_diff, save_baseline, load_baseline
    from calibrate_closed_loop import slow_move_to_joints

    with open(POINTS_FILE) as f:
        pts = json.load(f)
    for need in ("homeplace0", "pickhome1"):
        if need not in pts:
            print(f"ERROR: {need} missing from teach_points.json")
            sys.exit(1)

    q = QArmDriver(); q.connect(); time.sleep(0.4)
    for _ in range(5):
        try: q.read_all(); break
        except Exception: time.sleep(0.3)
    try:
        # 1. Move to homeplace0 (arm out of UGreen view).
        print("\n[1/4] Moving to homeplace0 ...")
        target = np.array(pts["homeplace0"]["joints_rad"], dtype=float)
        grip = float(pts["homeplace0"].get("gripper", 0.15))
        slow_move_to_joints(q, target, grip, seconds=3.0)
        time.sleep(0.5)

        # 2. Capture baseline.
        print("[2/4] Capturing UGreen baseline ...")
        baseline = capture(warmup=20)
        save_baseline(baseline, BASELINE_PNG)
        print(f"     -> {BASELINE_PNG}  ({baseline.shape} mean={baseline.mean():.0f})")

        # 3. Move to pickhome1.
        print("[3/4] Moving to pickhome1 ...")
        target = np.array(pts["pickhome1"]["joints_rad"], dtype=float)
        grip = float(pts["pickhome1"].get("gripper", 0.15))
        slow_move_to_joints(q, target, grip, seconds=3.0)
        time.sleep(0.5)

        # 4. Capture reference + TCP.
        print("[4/4] Capturing pickhome1 reference ...")
        ref = capture(warmup=20)
        cv2.imwrite(REF_PNG, ref)
        tcp = tcp_from_diff(ref, load_baseline(BASELINE_PNG))
        if tcp is None:
            print("WARN: no TCP detected at pickhome1 — baseline may not "
                  "have been clear enough, or arm silhouette too small.")
            sys.exit(2)
        with open(REF_TCP, "w") as f:
            json.dump({"tcp": list(tcp), "pose": "pickhome1"}, f, indent=2)
        print(f"     -> {REF_PNG}  TCP pixel = {tcp}")
        print(f"     -> {REF_TCP}")
    finally:
        try: q.card.close()
        except Exception: pass
        q._connected = False

    print("\nDone. Run python/preflight.py to confirm 7/7 green.")


if __name__ == "__main__":
    main()
