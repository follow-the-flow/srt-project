"""
Single-point calibration patch for session_cal.json.

If only the arm base shifted (homography + survey pose unchanged), every
reported base_m is wrong by the same translation. This script measures
that translation from ONE physical touch and applies it to
chess_origin_in_base_m.

Usage:
    py -3.13 python/fix_cal_single_point.py --reported X,Y,Z

Where X,Y,Z is the detector-reported base_m of a fruit you can physically
touch (e.g. the tomato from the last diag_detector run).

Flow:
    1. Connect arm.
    2. Jog gripper tip to the ACTUAL CENTER TOP of that same fruit.
    3. Press ENTER. Script reads FK -> actual base_m.
    4. delta = actual - reported.
    5. session_cal.json chess_origin_in_base_m += delta (backed up first).
"""
import argparse, json, os, shutil, sys, datetime
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from session_cal import SessionCal
from touch_probe import jog_and_capture

REPO = os.path.dirname(_HERE)
CAL_PATH = os.path.join(REPO, "session_cal.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reported", required=True,
        help="Detector-reported base_m of a fruit, e.g. '0.487,0.214,0.067'")
    args = ap.parse_args()
    reported = np.array([float(v) for v in args.reported.split(",")],
                         dtype=float)
    if reported.shape != (3,):
        sys.exit("--reported must be X,Y,Z")

    cal = SessionCal.load(CAL_PATH)
    print(f"  current chess_origin_in_base_m = "
          f"{cal.chess_origin_in_base_m.round(4).tolist()}")
    print(f"  reported target base_m         = "
          f"{reported.round(4).tolist()}")
    print()
    print("Jog the GRIPPER TIP to the PHYSICAL CENTER TOP of the same fruit")
    print("(same one the detector reported). Press ENTER to capture, ESC to")
    print("abort.")

    from qarm_driver import QArmDriver
    driver = QArmDriver()
    driver.connect()
    try:
        actual = jog_and_capture(driver)
    finally:
        try: driver.card.close()
        except Exception: pass
        driver._connected = False

    actual = np.asarray(actual, dtype=float)
    delta = actual - reported
    print(f"\n  actual jogged TCP     = {actual.round(4).tolist()}")
    print(f"  delta (actual-report) = {delta.round(4).tolist()}")
    print(f"  |delta|               = {np.linalg.norm(delta)*1000:.1f} mm")

    if np.linalg.norm(delta) > 0.10:
        print("\n[WARN] delta > 10 cm — almost certainly the wrong fruit or a")
        print("       bad jog. NOT patching. Re-run if needed.")
        return 1

    # Backup + patch
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = CAL_PATH + f".bak_{ts}"
    shutil.copy2(CAL_PATH, backup)
    print(f"\n  backup -> {backup}")

    new_origin = cal.chess_origin_in_base_m + delta
    cal.chess_origin_in_base_m = new_origin
    cal.timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    cal.save(CAL_PATH)
    print(f"  patched chess_origin_in_base_m = {new_origin.round(4).tolist()}")
    print(f"  written to {CAL_PATH}")
    print("\n  run diag_detector.py or diag_pick_one.py to verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
