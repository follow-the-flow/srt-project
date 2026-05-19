"""
Data carrier + JSON serialisation for the per-session calibration output
of calibrate_chessboard.py.

Consumed by survey_capture.py (built D2) and the fruit detector (D2).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from typing import Any
import numpy as np


@dataclass
class SessionCal:
    timestamp: str
    chess_origin_in_base_m: np.ndarray           # shape (3,)
    h_pixel_to_chess_mm: np.ndarray              # shape (3, 3)
    survey_pose_joints_rad: np.ndarray           # shape (4,)
    chess_pattern: dict                           # cols/rows/square_mm/...
    d415_intrinsics: dict                         # fx/fy/cx/cy
    homography_reproj_rms_px: float
    camera_height_above_table_m: float
    image_size: tuple                             # (w, h)
    # Added 2026-04-24: {"R_cam_in_base": 3x3 list, "C_cam_in_base_m": 3 list,
    # "reproj_rms_px": float} — None before first solvePnP run.
    cam_extrinsics_survey1: dict | None = None

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "chess_origin_in_base_m":
                np.asarray(self.chess_origin_in_base_m).tolist(),
            "H_pixel_to_chess_mm":
                np.asarray(self.h_pixel_to_chess_mm).tolist(),
            "survey_pose_joints_rad":
                np.asarray(self.survey_pose_joints_rad).tolist(),
            "chess_pattern": self.chess_pattern,
            "d415_intrinsics": self.d415_intrinsics,
            "homography_reproj_rms_px": float(self.homography_reproj_rms_px),
            "camera_height_above_table_m":
                float(self.camera_height_above_table_m),
            "image_size": list(self.image_size),
            "cam_extrinsics_survey1": self.cam_extrinsics_survey1,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SessionCal":
        with open(path, "r") as f:
            d = json.load(f)
        return cls(
            timestamp=d["timestamp"],
            chess_origin_in_base_m=np.asarray(d["chess_origin_in_base_m"]),
            h_pixel_to_chess_mm=np.asarray(d["H_pixel_to_chess_mm"]),
            survey_pose_joints_rad=np.asarray(d["survey_pose_joints_rad"]),
            chess_pattern=d["chess_pattern"],
            d415_intrinsics=d["d415_intrinsics"],
            homography_reproj_rms_px=float(d["homography_reproj_rms_px"]),
            camera_height_above_table_m=float(d["camera_height_above_table_m"]),
            image_size=tuple(d["image_size"]),
            cam_extrinsics_survey1=d.get("cam_extrinsics_survey1"),
        )
