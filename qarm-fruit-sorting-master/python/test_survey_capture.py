"""Tests for survey_capture helpers."""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from survey_capture import _warmup_and_capture, _chessboard_residual


class FakeCamera:
    """Returns `dark_count` dark frames, then bright frames forever."""
    def __init__(self, dark_count=3, img_shape=(720, 1280, 3)):
        self.dark_count = dark_count
        self.img_shape = img_shape
        self._reads = 0
        self.intrinsics = {"fx": 912.6, "fy": 911.3,
                            "cx": 635.4, "cy": 343.0}

    def read(self):
        self._reads += 1
        if self._reads <= self.dark_count:
            return (np.zeros(self.img_shape, dtype=np.uint8),
                    np.zeros(self.img_shape[:2], dtype=np.uint16))
        # Bright frame: uniform value 100 + some variation
        c = np.full(self.img_shape, 100, dtype=np.uint8)
        d = np.full(self.img_shape[:2], 300, dtype=np.uint16)
        return c, d


def test_warmup_skips_dark_frames():
    name = "warmup_skips_dark"
    cam = FakeCamera(dark_count=3)
    color, depth, intr = _warmup_and_capture(cam, warmup_timeout_s=2.0)
    assert color.shape == (720, 1280, 3), f"shape {color.shape}"
    assert color.mean() > 90, f"mean {color.mean()}"
    assert intr["fx"] == 912.6
    # dark_count dark warm-ups + 1 bright warm-up that breaks loop
    # + 5 capture reads = 9 total
    assert cam._reads == 9, f"reads {cam._reads}"
    return name, True, f"warmed up past {cam.dark_count} dark frames"


def test_warmup_timeout_raises():
    name = "warmup_timeout"
    # dark_count very high, timeout short -> must raise
    cam = FakeCamera(dark_count=10_000)
    try:
        _warmup_and_capture(cam, warmup_timeout_s=0.2)
    except RuntimeError as ex:
        return name, "warm-up timeout" in str(ex).lower(), str(ex)
    return name, False, "expected RuntimeError, got none"


def test_residual_no_chessboard_returns_none():
    """Plain grey image -> findChessboardCorners fails -> (None, 0)."""
    name = "residual_no_chessboard"
    from session_cal import SessionCal
    sc = SessionCal(
        timestamp="2026-04-23T00:00:00",
        chess_origin_in_base_m=np.array([0.4, 0.1, 0.0]),
        h_pixel_to_chess_mm=np.eye(3),
        survey_pose_joints_rad=np.zeros(4),
        chess_pattern={"cols": 8, "rows": 6, "square_mm": 30.0,
                       "inner_cols": 7, "inner_rows": 5},
        d415_intrinsics={"fx": 912.6, "fy": 911.3,
                          "cx": 635.4, "cy": 343.0},
        homography_reproj_rms_px=0.5,
        camera_height_above_table_m=0.26,
        image_size=(1280, 720),
    )
    grey = np.full((720, 1280, 3), 128, dtype=np.uint8)
    residual, n = _chessboard_residual(grey, sc)
    assert residual is None and n == 0, f"got ({residual}, {n})"
    return name, True, "no corners -> (None, 0)"


TESTS = [test_warmup_skips_dark_frames,
         test_warmup_timeout_raises,
         test_residual_no_chessboard_returns_none]


def test_capture_fruits_raises_when_arm_not_settled(monkeypatch):
    """If read_all returns joints far from the target survey pose after
    the 1.5 s dwell, capture_fruits must raise RuntimeError."""
    import numpy as np
    import pytest
    from survey_capture import capture_fruits
    from session_cal import SessionCal

    class StuckDriver:
        def read_all(self):
            # Returns joints that differ from survey by 0.2 rad on joint 2.
            return np.array([0.0, 0.5, 0.9, 0.0]), 0.1

    class DummyCam:
        intrinsics = {"fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0}
        def read(self):
            import numpy as np
            return np.full((480, 640, 3), 100, dtype=np.uint8), \
                   np.full((480, 640), 300, dtype=np.uint16)

    cal = SessionCal(
        timestamp="t",
        chess_origin_in_base_m=np.zeros(3),
        h_pixel_to_chess_mm=np.eye(3),
        survey_pose_joints_rad=np.array([0.0, 0.5, 0.7, 0.0]),
        chess_pattern={"cols": 7, "rows": 5, "square_mm": 30.0},
        d415_intrinsics={"fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0},
        homography_reproj_rms_px=0.5,
        camera_height_above_table_m=0.30,
        image_size=(640, 480),
    )

    # Stub out slow_move_to_joints (otherwise test would try to import driver).
    import survey_capture
    monkeypatch.setattr(survey_capture, "_settle_sleep_s", 0.01)
    monkeypatch.setattr(
        "calibrate_closed_loop.slow_move_to_joints",
        lambda *_args, **_kw: None,
    )

    with pytest.raises(RuntimeError, match="settle"):
        capture_fruits(StuckDriver(), DummyCam(), cal)


def main():
    fails = 0
    for fn in TESTS:
        try:
            name, ok, msg = fn()
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}  {msg}")
            if not ok: fails += 1
        except Exception as ex:
            import traceback
            fails += 1
            print(f"  [FAIL] {fn.__name__}: {ex}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(main())
