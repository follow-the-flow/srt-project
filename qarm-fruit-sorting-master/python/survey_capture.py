"""
One-shot survey capture: move to survey1, grab RGB+depth, self-check
chessboard residual, run detect_fruits.

See docs/superpowers/specs/2026-04-22-overhead-vision-pipeline-design.md §3.4.
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from session_cal import SessionCal
from fruit_detector import detect_fruits

_INNER_COLS = 7
_INNER_ROWS = 5
_SQUARE_MM = 30.0
_WARN_RESIDUAL_MM = 3.0
_ERROR_RESIDUAL_MM = 10.0

_SETTLE_SLEEP_S = 1.5           # canonical production dwell; do not reassign
_SETTLE_JOINT_TOL_RAD = 0.04    # relaxed 2026-04-27 in lab: at the
                                # current survey1 pose the QArm position-
                                # mode PID hits a steady-state error of
                                # ~0.025 rad (gravity load, no I term),
                                # so the previous 0.015 was unreachable
                                # and aborted every capture. 0.04 admits
                                # ~16 mm worst-case TCP drift; the
                                # downstream chessboard residual check
                                # (3 mm warn, 10 mm error) is the real
                                # mm-level gate.
# Runtime uses `_settle_sleep_s` so tests can monkeypatch it to 0.01.
_settle_sleep_s = _SETTLE_SLEEP_S


def _warmup_and_capture(camera, warmup_timeout_s: float = 10.0):
    """Poll camera.read() until color.mean()>5, then median 5 frames.

    Returns (color_bgr: HxWx3 uint8, depth_mm: HxW uint16, intrinsics: dict).
    Raises RuntimeError on timeout.
    """
    deadline = time.time() + warmup_timeout_s
    while time.time() < deadline:
        try:
            c, _ = camera.read()
        except Exception:
            continue
        if c.mean() > 5:
            break
    else:
        raise RuntimeError(
            "camera warm-up timeout - no valid frame after "
            f"{warmup_timeout_s:.0f} s.")
    frames = []
    depth_latest = None
    for _ in range(5):
        c, d = camera.read()
        frames.append(c.copy())
        depth_latest = d.copy() if d is not None else None
    color = np.median(np.stack(frames), axis=0).astype(np.uint8)
    intr = {k: float(camera.intrinsics[k]) for k in ("fx", "fy", "cx", "cy")}
    return color, depth_latest, intr


def _chessboard_residual(color_bgr: np.ndarray, session_cal: SessionCal):
    """Try to re-find chessboard corners in the live frame; project stored
    homography to image and compute per-corner delta in mm.

    Returns (residual_mm: float | None, corners_found: int).
    None if corners not detected - caller decides how to warn.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (_INNER_COLS, _INNER_ROWS),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None, 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    refined = cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1), criteria
    ).reshape(-1, 2).astype(np.float32)
    world_mm = np.array(
        [[i * _SQUARE_MM, j * _SQUARE_MM]
         for j in range(_INNER_ROWS) for i in range(_INNER_COLS)],
        dtype=np.float32)
    H = np.asarray(session_cal.h_pixel_to_chess_mm, dtype=np.float64)
    pts_h = np.hstack([refined, np.ones((refined.shape[0], 1), dtype=np.float32)])
    proj = (H @ pts_h.T).T
    proj_mm = proj[:, :2] / proj[:, 2:3]
    err_mm = np.linalg.norm(proj_mm - world_mm, axis=1)
    return float(np.sqrt(np.mean(err_mm ** 2))), int(refined.shape[0])


def capture_fruits(driver, camera, session_cal: SessionCal):
    """
    Survey-capture orchestrator.

    1. slow_move_to_joints(driver, survey_pose)
    2. warm up + capture RGB+depth
    3. compute chessboard residual (self-check)
    4. run detect_fruits

    Parameters
    ----------
    driver : QArmDriver  (already connected)
    camera : QArmCamera  (already opened)
    session_cal : SessionCal

    Returns
    -------
    detections : list[Detection]
    diagnostics : dict with keys
        chessboard_residual_mm (float | None)
        chessboard_corners_found (int)
        warnings (list[str])

    Raises
    ------
    RuntimeError if residual > 10 mm (calibration broken).
    """
    from calibrate_closed_loop import slow_move_to_joints  # existing helper
    slow_move_to_joints(driver, session_cal.survey_pose_joints_rad,
                        float(0.10))
    # Dwell so the position-mode PID can actually reach the target
    # (it has ~0.5-1.0 s tracking lag on big descents).
    time.sleep(_settle_sleep_s)
    # Sanity: confirm the arm actually got there; abort loudly if not.
    measured_joints, _ = driver.read_all()
    err = np.linalg.norm(
        np.asarray(measured_joints, dtype=float)
        - np.asarray(session_cal.survey_pose_joints_rad, dtype=float))
    if err > _SETTLE_JOINT_TOL_RAD:
        raise RuntimeError(
            f"arm failed to settle at survey1 "
            f"(joint-norm err {err:.3f} rad > {_SETTLE_JOINT_TOL_RAD})")
    color, depth, _ = _warmup_and_capture(camera)
    residual_mm, n_corners = _chessboard_residual(color, session_cal)
    warnings: list[str] = []
    if residual_mm is None:
        warnings.append("chessboard not detected in survey frame - "
                         "residual self-check skipped")
    else:
        if residual_mm > _ERROR_RESIDUAL_MM:
            raise RuntimeError(
                f"chessboard residual {residual_mm:.1f} mm > "
                f"{_ERROR_RESIDUAL_MM:.0f} mm - rerun "
                "calibrate_chessboard.py")
        if residual_mm > _WARN_RESIDUAL_MM:
            warnings.append(
                f"chessboard residual {residual_mm:.1f} mm > "
                f"{_WARN_RESIDUAL_MM:.0f} mm - drift detected")
    detections = detect_fruits(color, depth, session_cal)
    diagnostics = {
        "chessboard_residual_mm": residual_mm,
        "chessboard_corners_found": n_corners,
        "warnings": warnings,
        "color_frame": color,   # for the viewer to render
    }
    return detections, diagnostics
