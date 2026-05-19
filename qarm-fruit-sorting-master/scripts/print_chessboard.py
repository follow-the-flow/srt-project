"""Generate a 7x5 inner-corners chessboard PNG for UGreen intrinsic
calibration. Print on A4 at 100% scale; each square is 30 mm.
The printed pattern is the PHYSICAL reference — the user shows it to
the UGreen camera in 15+ orientations and `ugreen_intrinsics.calibrate`
extracts fx/fy/cx/cy + distortion."""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "chessboard_7x5_30mm.png")

# OpenCV findChessboardCorners expects INNER corners.
# 7x5 inner corners = 8x6 squares. Each square 30 mm.
inner = (7, 5)
squares = (inner[0] + 1, inner[1] + 1)   # 8 x 6
square_mm = 30.0
dpi = 300
mm_per_inch = 25.4
px_per_mm = dpi / mm_per_inch
sq_px = int(round(square_mm * px_per_mm))
W = squares[0] * sq_px
H = squares[1] * sq_px

img = np.full((H, W), 255, dtype=np.uint8)
for i in range(squares[0]):
    for j in range(squares[1]):
        if (i + j) % 2 == 0:
            x0 = i * sq_px
            y0 = j * sq_px
            img[y0:y0+sq_px, x0:x0+sq_px] = 0

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, img)
print(f"wrote {OUT}  ({W}x{H}px = "
      f"{W / px_per_mm:.0f}x{H / px_per_mm:.0f} mm, fits A4 landscape)")
print(f"Print at 100% scale, verify one square measures {square_mm:.0f} mm")
