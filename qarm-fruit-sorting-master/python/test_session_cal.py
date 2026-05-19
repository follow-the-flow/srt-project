"""Round-trip SessionCal through JSON with cam_extrinsics_survey1."""
from __future__ import annotations
import os
import sys
import json
import tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from session_cal import SessionCal


def _base_cal() -> SessionCal:
    return SessionCal(
        timestamp="2026-04-24T12:00:00",
        chess_origin_in_base_m=np.array([0.30, 0.00, 0.00]),
        h_pixel_to_chess_mm=np.eye(3),
        survey_pose_joints_rad=np.array([0.0, 0.5, 0.7, 0.0]),
        chess_pattern={"cols": 7, "rows": 5, "square_mm": 30.0},
        d415_intrinsics={"fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0},
        homography_reproj_rms_px=0.5,
        camera_height_above_table_m=0.254,
        image_size=(640, 480),
    )


def test_roundtrip_without_extrinsics():
    c = _base_cal()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        c.save(path)
        loaded = SessionCal.load(path)
        assert loaded.cam_extrinsics_survey1 is None
    finally:
        os.unlink(path)


def test_roundtrip_with_extrinsics():
    c = _base_cal()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.3])
    c.cam_extrinsics_survey1 = {
        "R_cam_in_base": R.tolist(),
        "C_cam_in_base_m": t.tolist(),
        "reproj_rms_px": 0.42,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        c.save(path)
        loaded = SessionCal.load(path)
        assert loaded.cam_extrinsics_survey1 is not None
        np.testing.assert_allclose(
            np.asarray(loaded.cam_extrinsics_survey1["R_cam_in_base"]), R)
        np.testing.assert_allclose(
            np.asarray(loaded.cam_extrinsics_survey1["C_cam_in_base_m"]), t)
        assert loaded.cam_extrinsics_survey1["reproj_rms_px"] == 0.42
    finally:
        os.unlink(path)
