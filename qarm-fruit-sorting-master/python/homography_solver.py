"""
Homography math for the chessboard session-calibration.

Pure OpenCV, no hardware imports. Kept separate from calibrate_chessboard.py
so the math is unit-testable against synthetic images.
"""
from __future__ import annotations
import numpy as np
import cv2


def solve_homography(image_pts: np.ndarray,
                     world_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve H mapping image pixels to world-XY (mm), plus the reprojection
    RMS in mm.

    Parameters
    ----------
    image_pts : (N, 2) float array of pixel coordinates.
    world_pts : (N, 2) float array of corresponding world coordinates.

    Returns
    -------
    H : (3, 3) float64. Multiply pixel (u, v, 1).T by H to get world (X, Y, 1).T.
    rms_mm : float. Root-mean-square reprojection error in world-mm.

    Raises ValueError if findHomography fails (e.g. collinear input, <4 pts).
    """
    image_pts = np.asarray(image_pts, dtype=np.float32)
    world_pts = np.asarray(world_pts, dtype=np.float32)
    if image_pts.shape[0] < 4:
        raise ValueError(f"need >=4 points, got {image_pts.shape[0]}")

    H, mask = cv2.findHomography(image_pts, world_pts, method=0)
    if H is None:
        raise ValueError("findHomography returned None (collinear input?)")

    proj = cv2.perspectiveTransform(
        image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - world_pts, axis=1)
    rms_mm = float(np.sqrt(np.mean(err ** 2)))
    return H.astype(np.float64), rms_mm


def camera_height_from_homography(H: np.ndarray,
                                   fx: float, fy: float,
                                   cx: float, cy: float) -> float:
    """
    Estimate camera height above the chess plane, in the same units as
    H's world range (so: mm if H maps pixel->mm).

    Derivation: for a nadir pinhole camera, the Jacobian of the pixel->world
    map near the principal point has determinant (h/fx)*(h/fy), so
        h = sqrt(|det J| * fx * fy).

    Accurate to ~2% for a nadir camera; ~15% for tilts up to ~30 deg
    (spec §4.3 enforces this). For larger tilts the value is only an
    approximation; the caller should not trust it past 20% rel err.
    """
    p = np.array([cx, cy, 1.0])
    w = H @ p
    s = w[2]
    if abs(s) < 1e-9:
        raise ValueError("homography degenerate at principal point")

    x = w[0] / s
    y = w[1] / s
    dxdu = (H[0, 0] - x * H[2, 0]) / s
    dxdv = (H[0, 1] - x * H[2, 1]) / s
    dydu = (H[1, 0] - y * H[2, 0]) / s
    dydv = (H[1, 1] - y * H[2, 1]) / s
    det_j = abs(dxdu * dydv - dxdv * dydu)
    h_est = float(np.sqrt(det_j * fx * fy))
    return h_est
