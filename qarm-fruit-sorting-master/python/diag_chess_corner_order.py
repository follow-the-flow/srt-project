"""
Diagnose whether OpenCV's corner-0 matches the user's 'top-left' jog.

Loads logs/calibration_latest.png, re-runs findChessboardCorners, and
annotates the four outermost corners + their world-frame coordinates so
you can see visually whether the stored H_pixel_to_chess_mm treats
your jogged corner as (0,0).
"""
import os, sys
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

repo = os.path.dirname(_HERE)
img_path = os.path.join(repo, "logs", "calibration_latest.png")
img = cv2.imread(img_path)
if img is None:
    sys.exit(f"missing {img_path}")

INNER_COLS, INNER_ROWS = 7, 5
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCorners(
    gray, (INNER_COLS, INNER_ROWS),
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
if not found:
    sys.exit("findChessboardCorners failed on calibration_latest.png")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
corners = corners.reshape(-1, 2)
print(f"found {corners.shape[0]} corners")

# The four "outermost" corners in detection order:
idx_map = {
    "corner[0]  (world 0,0)":                   0,
    f"corner[{INNER_COLS-1}]  (world {(INNER_COLS-1)*30},0)":      INNER_COLS - 1,
    f"corner[{INNER_COLS*(INNER_ROWS-1)}]  (world 0,{(INNER_ROWS-1)*30})":  INNER_COLS * (INNER_ROWS - 1),
    f"corner[{INNER_COLS*INNER_ROWS-1}]  (world {(INNER_COLS-1)*30},{(INNER_ROWS-1)*30})": INNER_COLS * INNER_ROWS - 1,
}
colors = [(0, 0, 255),    # red   = corner 0 (your jogged origin)
          (0, 165, 255),  # orange
          (0, 255, 255),  # yellow
          (0, 255, 0)]    # green = diagonally opposite

out = img.copy()
for (label, i), col in zip(idx_map.items(), colors):
    u, v = corners[i]
    cv2.circle(out, (int(u), int(v)), 16, col, 3)
    cv2.putText(out, label, (int(u) + 20, int(v) - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    print(f"  {label:55s}  pixel=({u:6.1f}, {v:6.1f})")

out_path = os.path.join(repo, "logs", "debug_chess_corner_order.png")
cv2.imwrite(out_path, out)
print(f"\nsaved -> {out_path}")
print("\nInterpretation:")
print("  RED circle  = corner[0] = world (0, 0) = where calibrate_chessboard")
print("                 thinks your jog-touched origin sits.")
print("  If the RED circle is NOT at the same corner you physically jogged to,")
print("  then the chess frame is rotated 90/180 deg relative to your mental")
print("  model -- that's the source of the systematic pick offset.")
