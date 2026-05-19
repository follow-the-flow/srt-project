"""Jog-and-touch three chessboard inner corners in sequence, print
kinematic TCP for each, compute the chessboard's in-plane rotation in
base frame, and (optionally) inject it into session_cal.json so
solvePnP gets a correctly-oriented 3D grid.

Run from repo root:
    py -3.13 python/touch_three_corners.py
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qarm_driver import QArmDriver
from touch_probe import jog_and_capture

_INNER_COLS = 7   # 7 corners across the long dim = i index 0..6
_INNER_ROWS = 5   # 5 corners across the short dim = j index 0..4
_SQUARE_MM = 30.0


def main():
    q = QArmDriver(); q.connect(); time.sleep(0.3)
    try:
        print("\n=== TOUCH 3 CHESSBOARD CORNERS ===")
        print("For each of 3 corners, jog the TCP tip onto the inner")
        print("corner (the black/white crossing), then press ENTER in")
        print("the touch-probe window.\n")

        print(">>> Corner 1 of 3: TOP-LEFT inner corner (i=0, j=0) <<<")
        tl = jog_and_capture(q)
        print(f"  TL TCP = {tl.round(4).tolist()}")

        print(">>> Corner 2 of 3: TOP-RIGHT inner corner (i=6, j=0)  "
              "— 6 squares along the LONG edge from TL <<<")
        tr = jog_and_capture(q)
        print(f"  TR TCP = {tr.round(4).tolist()}")

        print(">>> Corner 3 of 3: BOTTOM-LEFT inner corner (i=0, j=4) "
              "— 4 squares along the SHORT edge from TL <<<")
        bl = jog_and_capture(q)
        print(f"  BL TCP = {bl.round(4).tolist()}")

    finally:
        try: q.card.close()
        except Exception: pass
        q._connected = False

    # ----- compute chessboard axes in base frame -----
    tl = np.asarray(tl, dtype=float)
    tr = np.asarray(tr, dtype=float)
    bl = np.asarray(bl, dtype=float)

    i_vec = tr - tl     # along the 6-square long edge
    j_vec = bl - tl     # along the 4-square short edge
    i_len = np.linalg.norm(i_vec)
    j_len = np.linalg.norm(j_vec)

    print("\n=== CHESSBOARD GEOMETRY IN BASE FRAME ===")
    print(f"  TL->TR distance = {i_len*1000:.1f} mm "
          f"(expected {(_INNER_COLS-1)*_SQUARE_MM:.0f} mm for 6 squares)")
    print(f"  TL->BL distance = {j_len*1000:.1f} mm "
          f"(expected {(_INNER_ROWS-1)*_SQUARE_MM:.0f} mm for 4 squares)")

    i_axis = i_vec / i_len
    j_axis = j_vec / j_len
    print(f"  +i axis in base = {i_axis.round(4).tolist()}")
    print(f"  +j axis in base = {j_axis.round(4).tolist()}")

    # In-plane rotation of the board: angle between +i_axis and kinematic +x
    theta_rad = float(np.arctan2(i_axis[1], i_axis[0]))
    print(f"  board rotation = {np.degrees(theta_rad):+.2f} deg "
          "(angle between +i and kinematic +x)")

    dot = float(np.dot(i_axis, j_axis))
    print(f"  +i dot +j = {dot:.4f}  (should be ~0 if corners are orthogonal)")

    # Always persist to logs/chess_touched_corners.json. calibrate_chessboard.py
    # picks this up at Phase 1 and uses it to construct a correctly-oriented
    # 3D grid (bypassing the axis-aligned assumption).
    out_path = os.path.join(os.path.dirname(_HERE), "logs",
                              "chess_touched_corners.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "TL_base_m": tl.tolist(),
            "TR_base_m": tr.tolist(),
            "BL_base_m": bl.tolist(),
            "inner_cols": _INNER_COLS,
            "inner_rows": _INNER_ROWS,
            "square_mm": _SQUARE_MM,
            "i_axis_in_base": i_axis.tolist(),
            "j_axis_in_base": j_axis.tolist(),
            "board_rotation_deg": float(np.degrees(theta_rad)),
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    print("Now run: py -3.13 python/calibrate_chessboard.py --no-touch")
    print("(the --no-touch flag makes it reuse these 3 corners instead of "
          "prompting again)")


if __name__ == "__main__":
    main()
