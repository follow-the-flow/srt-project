"""
Unit + integration tests for the D1 chessboard-calibration subsystem.

Run with:
    py -3.13 python/test_calibrate_chessboard.py
Exit code is the number of failed sections (0 = all pass).
"""
import os
import sys
import json
import tempfile
import traceback
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_RESULTS = []


def _section(name, fn):
    try:
        fn()
        _RESULTS.append((name, True, None))
        print(f"  [PASS] {name}")
    except Exception as ex:
        _RESULTS.append((name, False, ex))
        print(f"  [FAIL] {name}: {ex}")
        traceback.print_exc()


# ========================================================================
# A1. session_cal IO
# ========================================================================
def test_session_cal_roundtrip():
    from session_cal import SessionCal

    cal = SessionCal(
        timestamp="2026-04-22T14:30:00",
        chess_origin_in_base_m=np.array([0.30, 0.10, 0.00]),
        h_pixel_to_chess_mm=np.eye(3),
        survey_pose_joints_rad=np.array([0.1, -0.2, 0.3, 0.0]),
        chess_pattern={"cols": 7, "rows": 5, "square_mm": 30,
                       "inner_cols": 6, "inner_rows": 4},
        d415_intrinsics={"fx": 912.6, "fy": 911.3,
                          "cx": 635.4, "cy": 343.0},
        homography_reproj_rms_px=0.45,
        camera_height_above_table_m=0.60,
        image_size=(1280, 720),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False) as f:
        path = f.name
    try:
        cal.save(path)
        roundtrip = SessionCal.load(path)
        assert np.allclose(roundtrip.chess_origin_in_base_m,
                            cal.chess_origin_in_base_m)
        assert np.allclose(roundtrip.h_pixel_to_chess_mm,
                            cal.h_pixel_to_chess_mm)
        assert roundtrip.timestamp == cal.timestamp
        assert roundtrip.chess_pattern["square_mm"] == 30
        assert roundtrip.image_size == (1280, 720)
    finally:
        os.unlink(path)


def test_session_cal_missing_file_raises():
    from session_cal import SessionCal
    try:
        SessionCal.load("/no/such/path.json")
        assert False, "should have raised"
    except FileNotFoundError:
        pass


# ========================================================================
# A2. homography_solver
# ========================================================================
def _make_synthetic_corners(inner_cols=7, inner_rows=5, square_mm=30,
                             fx=912.0, fy=912.0, cx=640.0, cy=360.0,
                             cam_height_mm=600.0):
    """Project the 35 inner chessboard corners onto a nadir pinhole
    camera of known intrinsics at cam_height_mm above the plane.
    Chess origin (0,0,0) is centred horizontally under the camera.
    Returns (image_pts (N,2) float32, world_pts (N,2) float32)."""
    pattern_w = (inner_cols - 1) * square_mm
    pattern_h = (inner_rows - 1) * square_mm
    x0, y0 = -pattern_w / 2, -pattern_h / 2
    world = np.array([[x0 + i * square_mm, y0 + j * square_mm]
                      for j in range(inner_rows)
                      for i in range(inner_cols)], dtype=np.float32)
    image = np.zeros_like(world)
    image[:, 0] = cx + fx * world[:, 0] / cam_height_mm
    image[:, 1] = cy + fy * world[:, 1] / cam_height_mm
    return image.astype(np.float32), world.astype(np.float32)


def test_homography_recovers_identity_scale():
    from homography_solver import solve_homography

    image, world = _make_synthetic_corners()
    H, rms = solve_homography(image, world)
    import cv2
    proj = cv2.perspectiveTransform(image.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - world, axis=1)
    assert err.max() < 0.1, f"max reprojection error {err.max():.4f} mm"
    assert rms < 0.05, f"rms {rms:.4f} too high"


def test_homography_rejects_collinear_points():
    from homography_solver import solve_homography
    collinear_img = np.array([[i * 10.0, 100.0] for i in range(10)],
                              dtype=np.float32)
    collinear_world = np.array([[i * 30.0, 0.0] for i in range(10)],
                                dtype=np.float32)
    try:
        solve_homography(collinear_img, collinear_world)
        assert False, "should have raised on collinear input"
    except ValueError:
        pass


# ========================================================================
# A3. camera height from homography
# ========================================================================
def test_camera_height_matches_synthetic_truth():
    from homography_solver import (solve_homography,
                                    camera_height_from_homography)
    true_h_mm = 600.0
    fx, fy, cx, cy = 912.0, 912.0, 640.0, 360.0
    image, world = _make_synthetic_corners(
        fx=fx, fy=fy, cx=cx, cy=cy, cam_height_mm=true_h_mm)
    H, _ = solve_homography(image, world)
    recovered = camera_height_from_homography(H, fx=fx, fy=fy, cx=cx, cy=cy)
    rel_err = abs(recovered - true_h_mm) / true_h_mm
    assert rel_err < 0.02, (
        f"recovered h {recovered:.1f} vs truth {true_h_mm:.1f} "
        f"(rel err {rel_err*100:.1f}%)")


def test_camera_height_handles_tilted_camera_within_15pct():
    """Approximate check: scale one axis by 0.94 (a rough stand-in for a
    mild tilt) and confirm the recovered height is within 15% of truth."""
    import cv2
    fx, fy, cx, cy = 912.0, 912.0, 640.0, 360.0
    true_h_mm = 600.0
    image, world = _make_synthetic_corners(
        fx=fx, fy=fy, cx=cx, cy=cy, cam_height_mm=true_h_mm)
    image_tilted = image.copy()
    image_tilted[:, 1] = (image_tilted[:, 1] - cy) * 0.94 + cy
    from homography_solver import (solve_homography,
                                    camera_height_from_homography)
    H, _ = solve_homography(image_tilted, world)
    recovered = camera_height_from_homography(H, fx=fx, fy=fy, cx=cx, cy=cy)
    rel_err = abs(recovered - true_h_mm) / true_h_mm
    assert rel_err < 0.15, (
        f"under mild tilt, recovered {recovered:.1f} vs {true_h_mm:.1f} "
        f"(rel err {rel_err*100:.1f}%)")


def test_camera_height_at_docstring_boundary_30deg():
    """Validate the docstring claim that ~30 deg tilt still stays inside
    the 15% tolerance. cos(30 deg) = 0.866, used as the y-axis scale."""
    import cv2
    fx, fy, cx, cy = 912.0, 912.0, 640.0, 360.0
    true_h_mm = 600.0
    image, world = _make_synthetic_corners(
        fx=fx, fy=fy, cx=cx, cy=cy, cam_height_mm=true_h_mm)
    image_tilted = image.copy()
    image_tilted[:, 1] = (image_tilted[:, 1] - cy) * 0.866 + cy
    from homography_solver import (solve_homography,
                                    camera_height_from_homography)
    H, _ = solve_homography(image_tilted, world)
    recovered = camera_height_from_homography(H, fx=fx, fy=fy, cx=cx, cy=cy)
    rel_err = abs(recovered - true_h_mm) / true_h_mm
    assert rel_err < 0.15, (
        f"at docstring boundary (30 deg, y-scale 0.866), recovered "
        f"{recovered:.1f} vs {true_h_mm:.1f} (rel err {rel_err*100:.1f}%)")


# ========================================================================
# A4. touch_probe
# ========================================================================
class _FakeDriver:
    """Stands in for QArmDriver. Stores last-commanded joints and
    returns them on read_all; ignores gripper commands."""
    def __init__(self, joints):
        self._j = np.asarray(joints, dtype=float)
        self._g = 0.10
    def read_all(self):
        return self._j.copy(), self._g
    def set_joints_and_gripper(self, j, g):
        self._j = np.asarray(j, dtype=float)
        self._g = float(g)


def test_touch_probe_returns_tcp_from_fk():
    """With the fake driver parked at a known pose, touch_probe.capture_tcp
    should return the FK position of that pose (no movement, just a read)."""
    from qarm_kinematics import forward_kinematics
    from touch_probe import capture_tcp

    joints = np.array([0.1, -0.2, 0.3, 0.0])
    driver = _FakeDriver(joints)
    expected_pos, _ = forward_kinematics(joints)
    tcp = capture_tcp(driver)
    assert np.allclose(tcp, expected_pos, atol=1e-9), (
        f"tcp {tcp} vs expected {expected_pos}")


def test_touch_probe_shape_is_3():
    from touch_probe import capture_tcp
    driver = _FakeDriver(np.array([0.0, 0.0, 0.0, 0.0]))
    tcp = capture_tcp(driver)
    assert tcp.shape == (3,), f"expected shape (3,) got {tcp.shape}"


# ========================================================================
# A5. calibrate_chessboard end-to-end (mocked hardware)
# ========================================================================
def _draw_synthetic_chessboard(inner_cols=7, inner_rows=5, square_px=80,
                                image_size=(1280, 720)):
    """Render a flat B/W chessboard centered in the image. Returns a BGR
    image suitable for cv2.findChessboardCorners."""
    import cv2
    W, H = image_size
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    ncol = inner_cols + 1
    nrow = inner_rows + 1
    pattern_w = ncol * square_px
    pattern_h = nrow * square_px
    x0 = (W - pattern_w) // 2
    y0 = (H - pattern_h) // 2
    for j in range(nrow):
        for i in range(ncol):
            if (i + j) % 2 == 1:
                x = x0 + i * square_px
                y = y0 + j * square_px
                img[y:y+square_px, x:x+square_px] = 0
    return img


def test_calibrate_chessboard_end_to_end_with_mocks():
    """Full pipeline with a fake driver + a rendered synthetic chessboard.
    Verifies the output session_cal.json is well-formed and the
    homography RMS is small."""
    import cv2
    from calibrate_chessboard import run_calibration_core
    from session_cal import SessionCal

    driver = _FakeDriver(np.array([0.1, -0.2, 0.3, 0.0]))
    frame = _draw_synthetic_chessboard()
    touched_tcp = np.array([0.30, 0.10, 0.00])
    survey_joints = np.array([0.0, 0.8, -0.5, 0.0])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False) as f:
        path = f.name
    try:
        run_calibration_core(
            touched_tcp=touched_tcp,
            survey_joints=survey_joints,
            captured_frame=frame,
            d415_intrinsics={"fx": 912.6, "fy": 911.3,
                              "cx": 635.4, "cy": 343.0},
            out_path=path,
        )
        cal = SessionCal.load(path)
        assert cal.homography_reproj_rms_px < 2.0, (
            f"rms {cal.homography_reproj_rms_px:.2f} px too high")
        assert cal.h_pixel_to_chess_mm.shape == (3, 3)
        assert np.allclose(cal.chess_origin_in_base_m, touched_tcp)
        assert np.allclose(cal.survey_pose_joints_rad, survey_joints)
        assert cal.chess_pattern["inner_cols"] == 7
        assert cal.chess_pattern["inner_rows"] == 5
        assert cal.camera_height_above_table_m > 0.1
    finally:
        os.unlink(path)


if __name__ == "__main__":
    _section("A1 session_cal roundtrip", test_session_cal_roundtrip)
    _section("A1 session_cal missing file", test_session_cal_missing_file_raises)
    _section("A2 homography recovers identity", test_homography_recovers_identity_scale)
    _section("A2 homography rejects collinear", test_homography_rejects_collinear_points)
    _section("A3 camera height nadir", test_camera_height_matches_synthetic_truth)
    _section("A3 camera height tilted", test_camera_height_handles_tilted_camera_within_15pct)
    _section("A3 camera height 30deg boundary", test_camera_height_at_docstring_boundary_30deg)
    _section("A4 touch_probe returns FK", test_touch_probe_returns_tcp_from_fk)
    _section("A4 touch_probe shape", test_touch_probe_shape_is_3)
    _section("A5 calibrate end-to-end",
             test_calibrate_chessboard_end_to_end_with_mocks)
    fails = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{len(_RESULTS)} test(s), {fails} failed")
    sys.exit(fails)
