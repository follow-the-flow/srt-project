"""
Diagnostic: have the operator jog the gripper TIP to the top center of
a fruit that the detector has reported. Compare jog TCP with reported
base_m to measure the residual error exactly.

Usage:
    py -3.13 python/diag_jog_to_fruit.py --reported X,Y,Z

X,Y,Z is the detector-reported base_m of the fruit you're jogging to.
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from touch_probe import jog_and_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reported", required=True,
        help="e.g. 0.618,0.089,0.069")
    args = ap.parse_args()
    reported = np.array([float(v) for v in args.reported.split(",")],
                         dtype=float)

    print(f"  reported target base_m = {reported.round(4).tolist()} m\n")
    print("Jog GRIPPER TIP to the PHYSICAL CENTER TOP of the same fruit.")
    print("Gripper pointing STRAIGHT DOWN. Press ENTER to capture.\n")

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
    dxy = np.hypot(delta[0], delta[1])
    print(f"\n  actual TCP = {actual.round(4).tolist()}")
    print(f"  reported   = {reported.round(4).tolist()}")
    print(f"  delta      = ({delta[0]*1000:+.1f}, {delta[1]*1000:+.1f}, "
          f"{delta[2]*1000:+.1f}) mm")
    print(f"  |delta_xy| = {dxy*1000:.1f} mm")
    print(f"  bearing    = {np.degrees(np.arctan2(delta[1], delta[0])):+.1f} deg "
          f"(+X=forward, +Y=robot's left)")
