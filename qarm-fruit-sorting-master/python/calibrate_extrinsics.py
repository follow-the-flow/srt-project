"""SolvePnP wrapper to recover D415 pose in base frame at survey1.
See docs/superpowers/specs/2026-04-24-accurate-pick-and-box-swap-design.md §3.1.
"""
from __future__ import annotations
import numpy as np
import cv2


def solve_survey1_extrinsics(corners_2d: np.ndarray,
                              corners_3d_base: np.ndarray,
                              K: np.ndarray,
                              dist_coeffs: np.ndarray,
                              chess_origin_z_in_base: float = 0.0):
    """Return (R_cam_in_base, C_cam_in_base_m, reproj_rms_px).

    Uses cv2.SOLVEPNP_IPPE (designed for planar targets) via
    solvePnPGeneric to get BOTH candidate solutions, then selects the one
    where the camera ends up above the chessboard plane
    (C_cam_in_base[2] > chess_origin_z_in_base). This resolves the
    classic planar-pose ambiguity that SOLVEPNP_ITERATIVE leaves open.

    Parameters
    ----------
    corners_2d : (N, 2) float32
    corners_3d_base : (N, 3) float32, base-frame corner coords
    K : (3, 3) intrinsics matrix
    dist_coeffs : (5,) or (4,)
    chess_origin_z_in_base : float, z of the chessboard plane in base
        frame (metres). Used only to pick the correct IPPE solution.

    Raises
    ------
    RuntimeError if IPPE fails, no above-plane solution exists, or the
    chosen solution's reprojection RMS > 5 px.
    """
    retval, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
        objectPoints=corners_3d_base.reshape(-1, 1, 3).astype(np.float32),
        imagePoints=corners_2d.reshape(-1, 1, 2).astype(np.float32),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist_coeffs.astype(np.float64),
        flags=cv2.SOLVEPNP_IPPE,
    )
    if retval < 1:
        raise RuntimeError("cv2.solvePnPGeneric(IPPE) returned no solutions")

    candidates = []   # list of (R_cam_in_base, C_cam_in_base, rms)
    for rvec, tvec in zip(rvecs, tvecs):
        R_base_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_in_base = R_base_to_cam.T
        C_cam_in_base = (-R_base_to_cam.T @ tvec).reshape(3)
        proj, _ = cv2.projectPoints(
            corners_3d_base.reshape(-1, 1, 3).astype(np.float32),
            rvec, tvec, K.astype(np.float64), dist_coeffs.astype(np.float64))
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - corners_2d.reshape(-1, 2), axis=1)
        rms = float(np.sqrt(np.mean(err ** 2)))
        candidates.append((R_cam_in_base, C_cam_in_base, rms))

    # Pick candidates with camera strictly above the chessboard plane.
    above = [c for c in candidates if c[1][2] > chess_origin_z_in_base]
    if not above:
        c_list = ", ".join(
            f"C={c[1].round(3).tolist()} rms={c[2]:.2f}" for c in candidates)
        raise RuntimeError(
            "no IPPE solution places the camera above the chessboard "
            f"plane (chess_origin_z={chess_origin_z_in_base:.3f}); "
            f"candidates: {c_list}")

    # Prefer smallest-RMS among above-plane solutions.
    R_cam_in_base, C_cam_in_base, rms = min(above, key=lambda c: c[2])

    if rms > 5.0:
        raise RuntimeError(
            f"solvePnP reprojection RMS {rms:.2f} px > 5 px; "
            "calibration rejected")
    return R_cam_in_base, C_cam_in_base, rms
