"""Load the saved calibration frame, run findChessboardCorners, and
annotate the four corner-grid extremes so the operator can identify
which physical corner OpenCV put at index 0."""
from __future__ import annotations
import os
import sys
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_INNER_COLS = 7
_INNER_ROWS = 5
_LOG_FRAME = os.path.join(
    os.path.dirname(_HERE), "logs", "calibration_latest.png")


def main():
    if not os.path.exists(_LOG_FRAME):
        sys.exit(f"no calibration frame at {_LOG_FRAME} - "
                  "rerun calibrate_chessboard.py first")
    bgr = cv2.imread(_LOG_FRAME)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (_INNER_COLS, _INNER_ROWS),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        sys.exit("findChessboardCorners failed on saved frame")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1), criteria
    ).reshape(-1, 2).astype(np.float32)

    # Four extreme indices on the 7x5 grid (row-major over rows then cols):
    extremes = {
        0:  ("image_pts[0]  (OpenCV's first corner)",        (0, 255, 255)),  # yellow
        _INNER_COLS - 1: ("image_pts[6]  (end of first row)",  (0, 255,   0)),  # green
        _INNER_COLS * (_INNER_ROWS - 1): ("image_pts[28] (start of last row)", (255, 0, 0)),  # blue
        _INNER_COLS * _INNER_ROWS - 1:  ("image_pts[34] (last corner)",       (0, 0, 255)),  # red
    }

    out = bgr.copy()
    # Draw all corners as small circles
    for p in corners:
        cv2.circle(out, (int(p[0]), int(p[1])), 4, (100, 100, 100), 1)

    print("OpenCV 4 extreme corners in image coordinates:")
    for idx, (label, color) in extremes.items():
        cx, cy = int(corners[idx][0]), int(corners[idx][1])
        print(f"  index {idx:2d}: pixel ({cx:4d}, {cy:4d})  colour ({color})")
        cv2.circle(out, (cx, cy), 18, color, 3)
        cv2.putText(out, f"[{idx}]", (cx + 22, cy + 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Save annotated image
    dst = os.path.join(
        os.path.dirname(_HERE), "logs", "corner_order_debug.png")
    cv2.imwrite(dst, out)
    print(f"\nAnnotated image saved to {dst}")
    print("Legend:")
    print("  yellow [0]  = OpenCV's first corner (image_pts[0])")
    print("  green  [6]  = end of first row in pattern space")
    print("  blue  [28]  = start of last row")
    print("  red   [34]  = last corner")


if __name__ == "__main__":
    main()
