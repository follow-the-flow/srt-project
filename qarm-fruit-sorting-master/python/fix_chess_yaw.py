"""
Rotate the stored H_pixel_to_chess_mm so its output is in BASE-XY, not in
the chessboard's own XY frame.

Fixes the systematic pick offset caused by calibrate_chessboard.py not
recording the rotation between chess and base frames (it only touches
corner[0] = origin; chess +X axis direction in base is never measured).

Flow
----
1. Load session_cal.json.
2. Prompt operator to jog gripper tip to corner[6] — the inner corner
   that is 180 mm along chess +X from corner[0] (i.e. the SAME ROW as
   your jogged origin, at the OTHER END of that row = 6 squares away).
3. Read TCP at corner[6] via FK.
4. Compute yaw = atan2 of (P6 - P0) in base XY.
5. Rotate H by yaw so pixel -> output is now base-XY delta (from origin).
6. Back up + rewrite session_cal.json.
"""
import datetime, os, shutil, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from session_cal import SessionCal
from touch_probe import jog_and_capture

REPO = os.path.dirname(_HERE)
CAL_PATH = os.path.join(REPO, "session_cal.json")


def main():
    cal = SessionCal.load(CAL_PATH)
    P0 = np.asarray(cal.chess_origin_in_base_m, dtype=float)
    print(f"  corner[0] origin (stored) = {P0.round(4).tolist()} m\n")
    print("Jog GRIPPER TIP to CORNER[6] — the inner corner SIX squares along")
    print("the FIRST ROW from where you touched origin. I.e. same edge of the")
    print("chessboard as your jogged origin, all the way to the OTHER end of")
    print("that top row (180 mm away from origin in chess +X).")
    print("Press ENTER to confirm, ESC to abort.\n")

    from qarm_driver import QArmDriver
    driver = QArmDriver()
    driver.connect()
    try:
        P6 = jog_and_capture(driver)
    finally:
        try: driver.card.close()
        except Exception: pass
        driver._connected = False
    P6 = np.asarray(P6, dtype=float)

    dx = P6[0] - P0[0]
    dy = P6[1] - P0[1]
    dist_mm = np.hypot(dx, dy) * 1000
    print(f"\n  corner[6] TCP = {P6.round(4).tolist()} m")
    print(f"  (P6 - P0)     = ({dx*1000:+.1f}, {dy*1000:+.1f}) mm  "
          f"|d|={dist_mm:.1f} mm (expected ~180)")
    if dist_mm < 150 or dist_mm > 220:
        print(f"\n[WARN] distance {dist_mm:.1f} mm is far from expected 180 mm.")
        print("       Did you jog to the RIGHT corner (7th inner corner on row 0)?")
        print("       Re-run if the distance is way off.")
        resp = input("  Proceed anyway? (y/N): ").strip().lower()
        if resp != "y":
            return 1

    # chess +X axis direction in base XY:
    yaw = float(np.arctan2(dy, dx))
    print(f"\n  chess +X axis angle in base frame = {np.degrees(yaw):+.2f} deg")

    # Rotation matrix that takes chess-XY (mm) -> base-XY delta (mm).
    # If chess +X sits at angle yaw in base, then a vector (1,0) in chess
    # maps to (cos yaw, sin yaw) in base -> R = [[cos,-sin],[sin,cos]].
    c, s = np.cos(yaw), np.sin(yaw)
    R3 = np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]], dtype=float)
    H_old = np.asarray(cal.h_pixel_to_chess_mm, dtype=float)
    H_new = R3 @ H_old
    print(f"\n  old H (pixel->chess-XY):")
    for row in H_old: print(f"    {row.tolist()}")
    print(f"  new H (pixel->base-XY delta):")
    for row in H_new: print(f"    {row.tolist()}")

    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = CAL_PATH + f".bak_{ts}"
    shutil.copy2(CAL_PATH, backup)
    print(f"\n  backup -> {backup}")

    cal.h_pixel_to_chess_mm = H_new
    cal.timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    cal.save(CAL_PATH)
    print(f"  session_cal.json updated. H now outputs base-XY delta directly.")
    print("\n  verify with: py -3.13 python/diag_detector.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
