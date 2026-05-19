"""
Unit tests for the rewritten fruit_detector.py (spec §3.2).

Run with:
    py -3.13 python/test_fruit_detector.py
Exit code is the number of failed sections (0 = all pass).
"""
import os
import sys
import traceback
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_RESULTS = []


def _nadir_extrinsics(height_m: float = 0.30, origin_m=(0.30, 0.0, 0.0)):
    """Build a cam_extrinsics_survey1 dict for a perfectly nadir camera
    above (origin_m.xy, origin_m.z + height_m). Used to keep legacy
    detector tests close to their old nadir-pinhole behaviour while still
    exercising the new ray-plane pipeline.

    R maps camera +Z (OpenCV convention: optical axis into scene) to
    base -Z (down toward the table).
    """
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]], dtype=np.float64)
    C = np.array([origin_m[0], origin_m[1], origin_m[2] + height_m],
                  dtype=np.float64)
    return {
        "R_cam_in_base": R.tolist(),
        "C_cam_in_base_m": C.tolist(),
        "reproj_rms_px": 0.1,
    }


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
# B1. Detection dataclass + module import
# ========================================================================
def test_detection_fields():
    from fruit_detector import Detection
    d = Detection(
        fruit_type="banana",
        center_px=(100, 200),
        center_base_m=np.array([0.3, 0.1, 0.03]),
        confidence=0.87,
        area_px=1500,
        bbox=(50, 150, 100, 80),
    )
    assert d.fruit_type == "banana"
    assert d.center_px == (100, 200)
    assert np.allclose(d.center_base_m, [0.3, 0.1, 0.03])
    assert d.confidence == 0.87
    assert d.area_px == 1500
    assert d.bbox == (50, 150, 100, 80)


def test_detection_to_dict():
    from fruit_detector import Detection
    d = Detection(
        fruit_type="tomato",
        center_px=(10, 20),
        center_base_m=np.array([0.1, 0.2, 0.04]),
        confidence=0.5,
        area_px=800,
        bbox=(5, 10, 20, 20),
    )
    out = d.to_dict()
    assert out["fruit_type"] == "tomato"
    assert out["center_px"] == [10, 20]
    assert out["center_base_m"] == [0.1, 0.2, 0.04]
    assert out["confidence"] == 0.5
    assert out["area_px"] == 800
    assert out["bbox"] == [5, 10, 20, 20]


# ========================================================================
# B2. hsv_mask helper
# ========================================================================
def _solid_bgr(color_bgr, size=(480, 640)):
    """Return a size-(H, W, 3) uint8 BGR image of a solid color."""
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def test_hsv_mask_single_range_yellow():
    from fruit_detector import hsv_mask
    yellow = _solid_bgr((0, 220, 220))  # BGR for bright yellow
    ranges = {"h": [18, 35], "s": [80, 255], "v": [80, 255]}
    mask = hsv_mask(yellow, ranges)
    assert mask.dtype == np.uint8
    assert mask.shape == (480, 640)
    assert (mask > 0).mean() > 0.95, (
        f"yellow mask coverage {(mask > 0).mean():.2f} too low")


def test_hsv_mask_red_wrap_around():
    from fruit_detector import hsv_mask
    red = _solid_bgr((0, 0, 220))
    ranges = {"h_wrap1": [0, 10], "h_wrap2": [170, 180],
              "s": [80, 255], "v": [60, 255]}
    mask = hsv_mask(red, ranges)
    assert (mask > 0).mean() > 0.95, (
        f"red mask coverage {(mask > 0).mean():.2f} too low")


def test_hsv_mask_excludes_out_of_range():
    from fruit_detector import hsv_mask
    blue = _solid_bgr((220, 0, 0))
    ranges = {"h": [18, 35], "s": [80, 255], "v": [80, 255]}
    mask = hsv_mask(blue, ranges)
    assert (mask > 0).mean() < 0.01


# ========================================================================
# B3. banana detection
# ========================================================================
def _draw_rect_bgr(size, center, w, h, color, rotation_deg=0):
    """Render a filled rotated rect on a black size=(H,W) canvas. Returns BGR."""
    import cv2
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    box = cv2.boxPoints(((center[0], center[1]), (w, h), rotation_deg))
    box = np.int32(box)
    cv2.fillPoly(img, [box], color)
    return img


def test_banana_contours_finds_one():
    from fruit_detector import _detect_banana_contours
    img = _draw_rect_bgr((480, 640), (320, 240), 180, 60,
                          (0, 220, 220))
    dets = _detect_banana_contours(img)
    assert len(dets) == 1, f"expected 1 banana, got {len(dets)}"
    (cx, cy), area, bbox, conf, _green_px = dets[0]
    assert abs(cx - 320) < 10, f"cx={cx}"
    assert abs(cy - 240) < 10, f"cy={cy}"
    assert area > 8000, f"area={area}"
    assert conf > 0.3, f"confidence={conf}"


def test_banana_rejects_round_shape():
    import cv2
    from fruit_detector import _detect_banana_contours
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 60, (0, 220, 220), -1)
    dets = _detect_banana_contours(img)
    assert len(dets) == 0, f"expected 0, got {len(dets)}"


def test_banana_rejects_tiny_blob():
    from fruit_detector import _detect_banana_contours
    img = _draw_rect_bgr((480, 640), (320, 240), 40, 10,
                          (0, 220, 220))
    dets = _detect_banana_contours(img)
    assert len(dets) == 0


# ========================================================================
# B4. tomato detection
# ========================================================================
def _draw_circle_bgr(size, center, r, color):
    import cv2
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.circle(img, center, r, color, -1)
    return img


def test_tomato_finds_round_red():
    from fruit_detector import _detect_tomato_contours
    img = _draw_circle_bgr((480, 640), (320, 240), 50, (0, 0, 220))
    dets = _detect_tomato_contours(img)
    assert len(dets) == 1, f"expected 1 tomato, got {len(dets)}"
    (cx, cy), area, bbox, conf, _green_px = dets[0]
    assert abs(cx - 320) < 5
    assert abs(cy - 240) < 5
    assert area > 7000, f"area={area}"
    assert conf > 0.5


def test_tomato_rejects_red_with_green_above():
    """A red circle with a green patch directly above it should be read
    as a strawberry candidate, not a tomato."""
    import cv2
    from fruit_detector import _detect_tomato_contours
    img = _draw_circle_bgr((480, 640), (320, 240), 50, (0, 0, 220))
    cv2.rectangle(img, (290, 170), (350, 185), (0, 200, 0), -1)
    dets = _detect_tomato_contours(img)
    assert len(dets) == 0, (
        f"tomato detector must reject red+green-above; got {len(dets)}")


def test_tomato_rejects_elongated_red():
    from fruit_detector import _detect_tomato_contours
    img = _draw_rect_bgr((480, 640), (320, 240), 160, 30,
                         (0, 0, 220))
    dets = _detect_tomato_contours(img)
    assert len(dets) == 0, "elongated red must not register as tomato"


# ========================================================================
# B5. strawberry detection
# ========================================================================
def _draw_strawberry_bgr(size, center, w, h_fruit, h_calyx=12):
    """A triangle-ish red body (wide top, narrow bottom) with a green band
    immediately above (the calyx). Good enough to exercise the taper score
    and the green-calyx confirm."""
    import cv2
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cx, cy = center
    top_w = w
    bot_w = w // 3
    pts = np.array([
        [cx - top_w // 2, cy - h_fruit // 2],
        [cx + top_w // 2, cy - h_fruit // 2],
        [cx + bot_w // 2, cy + h_fruit // 2],
        [cx - bot_w // 2, cy + h_fruit // 2],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 220))
    gy_lo = cy - h_fruit // 2 - 20
    gy_hi = cy - h_fruit // 2 - 6
    cv2.rectangle(img, (cx - top_w // 2, gy_lo),
                   (cx + top_w // 2, gy_hi), (0, 200, 0), -1)
    return img


def test_strawberry_finds_tapered_red_with_calyx():
    from fruit_detector import _detect_strawberry_contours
    img = _draw_strawberry_bgr((480, 640), (320, 240), 60, 70)
    dets = _detect_strawberry_contours(img)
    assert len(dets) == 1, f"expected 1 strawberry, got {len(dets)}"
    (cx, cy), area, bbox, conf, _green_px = dets[0]
    assert abs(cx - 320) < 10
    assert conf > 0.3


def test_strawberry_rejects_red_no_calyx():
    """Same tapered shape but no green above → not a strawberry."""
    import cv2
    from fruit_detector import _detect_strawberry_contours
    img = _draw_strawberry_bgr((480, 640), (320, 240), 60, 70)
    # Blank out the calyx band.
    img[180:240, 280:360] = 0
    cv2.fillPoly(img, [np.array([
        [290, 205], [350, 205], [340, 275], [300, 275]], dtype=np.int32)],
        (0, 0, 220))
    dets = _detect_strawberry_contours(img)
    assert len(dets) == 0, (
        f"strawberry must require green calyx; got {len(dets)}")


# ========================================================================
# B6. depth sampling
# ========================================================================
def test_depth_sample_returns_median():
    from fruit_detector import sample_depth_at_pixel
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[238:243, 318:323] = np.array([
        [100, 105, 110, 108, 102],
        [107, 113, 115, 112, 106],
        [104, 109, 120, 116, 108],
        [103, 108, 113, 110, 105],
        [101, 106, 111, 107, 102]], dtype=np.uint16)
    z = sample_depth_at_pixel(depth, (320, 240), patch=5)
    assert z is not None
    assert 105 < z < 115, f"median {z} outside expected"


def test_depth_sample_rejects_mostly_zero():
    from fruit_detector import sample_depth_at_pixel
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[238, 318:321] = 100
    z = sample_depth_at_pixel(depth, (320, 240), patch=5)
    assert z is None, f"should reject sparse depth, got {z}"


def test_depth_sample_rejects_out_of_range():
    from fruit_detector import sample_depth_at_pixel
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    z = sample_depth_at_pixel(depth, (320, 240), patch=5,
                               min_mm=10, max_mm=800)
    assert z is None


# ========================================================================
# B7. pixel -> base-frame projection
# ========================================================================
def _fake_session_cal(origin=(0.30, 0.10, 0.02),
                      cam_h_m=0.60, square_mm=30):
    """Build a SessionCal-like dict for testing projection math.
    H maps pixel (u,v) to chess-XY in mm assuming a nadir camera at
    cam_h_m above the chess plane."""
    from session_cal import SessionCal
    fx = 912.0
    fy = 912.0
    cx_p = 640.0
    cy_p = 360.0
    cam_h_mm = cam_h_m * 1000.0
    sx = cam_h_mm / fx
    sy = cam_h_mm / fy
    H = np.array([
        [sx,  0.0, -cx_p * sx],
        [0.0, sy,  -cy_p * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    cal = SessionCal(
        timestamp="2026-04-22T00:00:00",
        chess_origin_in_base_m=np.array(origin),
        h_pixel_to_chess_mm=H,
        survey_pose_joints_rad=np.zeros(4),
        chess_pattern={"cols": 8, "rows": 6, "square_mm": square_mm,
                        "inner_cols": 7, "inner_rows": 5},
        d415_intrinsics={"fx": fx, "fy": fy, "cx": cx_p, "cy": cy_p},
        homography_reproj_rms_px=0.5,
        camera_height_above_table_m=cam_h_m,
        image_size=(1280, 720),
        cam_extrinsics_survey1=_nadir_extrinsics(
            height_m=cam_h_m, origin_m=origin),
    )
    return cal


def test_pixel_to_base_identity_at_principal_point():
    from fruit_detector import pixel_to_base_frame
    cal = _fake_session_cal(origin=(0.30, 0.10, 0.02), cam_h_m=0.60)
    xyz = pixel_to_base_frame((640, 360), fruit_top_z_mm=0.0, session_cal=cal)
    assert np.allclose(xyz[:2], [0.30, 0.10], atol=1e-3), xyz
    assert abs(xyz[2] - 0.02) < 1e-6


def test_pixel_to_base_parallax_shrinks_with_fruit_height():
    """A pixel 100 px off the principal point with a tall fruit (50 mm)
    should land closer to the principal point's base XY than the same
    pixel with a flat fruit (0 mm). That is, parallax correction moves
    the reported XY TOWARDS the optical axis as the fruit gets taller
    (since we are reporting where the fruit BASE sits)."""
    from fruit_detector import pixel_to_base_frame
    cal = _fake_session_cal(origin=(0.30, 0.10, 0.02), cam_h_m=0.60)
    xyz_flat = pixel_to_base_frame((740, 360), fruit_top_z_mm=0.0,
                                    session_cal=cal)
    xyz_tall = pixel_to_base_frame((740, 360), fruit_top_z_mm=50.0,
                                    session_cal=cal)
    assert xyz_flat[0] > xyz_tall[0] > 0.30, (
        f"flat={xyz_flat} tall={xyz_tall}")


def test_pixel_to_base_z_equals_origin_plus_fruit_top():
    from fruit_detector import pixel_to_base_frame
    cal = _fake_session_cal(origin=(0.30, 0.10, 0.02), cam_h_m=0.60)
    xyz = pixel_to_base_frame((640, 360), fruit_top_z_mm=45.0,
                               session_cal=cal)
    assert abs(xyz[2] - (0.02 + 0.045)) < 1e-6, xyz


# ========================================================================
# B8. top-level detect_fruits
# ========================================================================
def _compose_test_scene():
    """Render a scene with one banana, one tomato, and one strawberry at
    known pixel locations. Return (bgr, depth, expected_pixels_by_type)."""
    import cv2
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.zeros((720, 1280), dtype=np.uint16)
    # Banana at (300, 360), yellow, elongated (150x60 gives aspect 2.5,
    # area 9000 — kept above _BANANA_MIN_AREA=5000 which was raised in
    # 2026-04-27 to drop yellowed strawberry-calyx false positives).
    cv2.rectangle(bgr, (225, 330), (375, 390), (0, 220, 220), -1)
    depth[320:400, 215:385] = 500
    # Tomato at (640, 360), round red.
    cv2.circle(bgr, (640, 360), 40, (0, 0, 220), -1)
    depth[315:405, 595:685] = 510
    # Strawberry at (960, 360), tapered red + calyx above.
    pts = np.array([[930, 320], [990, 320], [975, 400], [945, 400]],
                    dtype=np.int32)
    cv2.fillPoly(bgr, [pts], (0, 0, 220))
    cv2.rectangle(bgr, (930, 300), (990, 315), (0, 200, 0), -1)
    depth[310:410, 920:1000] = 505
    return bgr, depth, {
        "banana": (300, 360),
        "tomato": (640, 360),
        "strawberry": (960, 360),
    }


def test_detect_fruits_end_to_end():
    from fruit_detector import detect_fruits
    cal = _fake_session_cal(origin=(0.30, 0.10, 0.02), cam_h_m=0.60)
    bgr, depth, expected = _compose_test_scene()
    dets = detect_fruits(bgr, depth, cal)
    types = [d.fruit_type for d in dets]
    assert "banana" in types, f"no banana among {types}"
    assert "tomato" in types, f"no tomato among {types}"
    assert "strawberry" in types, f"no strawberry among {types}"
    for d in dets:
        assert d.center_base_m.shape == (3,)
        assert np.isfinite(d.center_base_m).all()
        assert 0.0 < d.confidence <= 1.0


def test_detect_fruits_falls_back_on_invalid_depth():
    """Fruit blobs with no valid depth should still be detected, using the
    per-type default height from _FRUIT_TOP_Z_MM. Dropping on null depth
    (the original spec Q5=B behaviour) was too strict for the tilted
    survey pose; D4 smoke test showed D415 depth was unreliable. We
    degrade gracefully to a flat-table assumption instead."""
    from fruit_detector import detect_fruits, _FRUIT_TOP_Z_MM
    cal = _fake_session_cal(origin=(0.30, 0.10, 0.02), cam_h_m=0.60)
    bgr, depth, _ = _compose_test_scene()
    depth = np.zeros_like(depth)
    dets = detect_fruits(bgr, depth, cal)
    types = [d.fruit_type for d in dets]
    assert "banana" in types
    assert "tomato" in types
    assert "strawberry" in types
    for d in dets:
        expected_z = cal.chess_origin_in_base_m[2] + _FRUIT_TOP_Z_MM[d.fruit_type] / 1000.0
        assert abs(d.center_base_m[2] - expected_z) < 1e-6, (
            f"{d.fruit_type}: base_z {d.center_base_m[2]:.4f} vs expected "
            f"{expected_z:.4f}")


def test_detect_fruits_empty_scene():
    from fruit_detector import detect_fruits
    cal = _fake_session_cal(origin=(0.30, 0.10, 0.02), cam_h_m=0.60)
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.full((720, 1280), 500, dtype=np.uint16)
    dets = detect_fruits(bgr, depth, cal)
    assert dets == []


def test_pixel_to_base_frame_uses_extrinsics_when_present():
    """With known extrinsics (camera directly above chess origin at 0.3 m,
    nadir), pixel = principal point + 0 should land at chess origin XY,
    and z should match fruit_top_z."""
    import numpy as np
    from session_cal import SessionCal
    from fruit_detector import pixel_to_base_frame

    K = {"fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0}
    origin = np.array([0.30, 0.00, 0.00])
    R_cam_in_base = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])  # camera looking down (+Z_cam = -Z_base)
    C_cam_in_base = np.array([0.30, 0.00, 0.30])

    cal = SessionCal(
        timestamp="t",
        chess_origin_in_base_m=origin,
        h_pixel_to_chess_mm=np.eye(3),
        survey_pose_joints_rad=np.zeros(4),
        chess_pattern={"cols": 7, "rows": 5, "square_mm": 30.0},
        d415_intrinsics=K,
        homography_reproj_rms_px=0.5,
        camera_height_above_table_m=0.30,
        image_size=(640, 480),
        cam_extrinsics_survey1={
            "R_cam_in_base": R_cam_in_base.tolist(),
            "C_cam_in_base_m": C_cam_in_base.tolist(),
            "reproj_rms_px": 0.1,
        },
    )
    # Pixel on the optical axis, fruit top 50 mm above the table.
    xyz = pixel_to_base_frame((320, 240), fruit_top_z_mm=50.0, session_cal=cal)
    np.testing.assert_allclose(xyz[:2], origin[:2], atol=1e-4)
    np.testing.assert_allclose(xyz[2], 0.050, atol=1e-4)


def test_pixel_to_base_frame_raises_when_extrinsics_missing():
    """Without extrinsics, projection must fail loudly; we will not
    silently fall back to the old nadir-pinhole formula."""
    import numpy as np
    import pytest
    from session_cal import SessionCal
    from fruit_detector import pixel_to_base_frame
    cal = SessionCal(
        timestamp="t",
        chess_origin_in_base_m=np.zeros(3),
        h_pixel_to_chess_mm=np.eye(3),
        survey_pose_joints_rad=np.zeros(4),
        chess_pattern={"cols": 7, "rows": 5, "square_mm": 30.0},
        d415_intrinsics={"fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0},
        homography_reproj_rms_px=0.5,
        camera_height_above_table_m=0.30,
        image_size=(640, 480),
    )
    with pytest.raises(ValueError):
        pixel_to_base_frame((320, 240), 50.0, cal)


def test_pixel_to_base_frame_offset_camera():
    """Non-nadir camera (5 cm lateral offset, 15° forward tilt) still
    projects a central pixel to a deterministic base point — verifies
    ray-plane math, not homography."""
    import numpy as np
    from session_cal import SessionCal
    from fruit_detector import pixel_to_base_frame

    K = {"fx": 380.0, "fy": 380.0, "cx": 320.0, "cy": 240.0}
    tilt = np.deg2rad(15.0)
    R_cam_in_base = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(np.pi + tilt), -np.sin(np.pi + tilt)],
        [0.0, np.sin(np.pi + tilt),  np.cos(np.pi + tilt)],
    ])
    C_cam_in_base = np.array([0.40, 0.05, 0.33])

    cal = SessionCal(
        timestamp="t",
        chess_origin_in_base_m=np.array([0.30, 0.00, 0.00]),
        h_pixel_to_chess_mm=np.eye(3),
        survey_pose_joints_rad=np.zeros(4),
        chess_pattern={"cols": 7, "rows": 5, "square_mm": 30.0},
        d415_intrinsics=K,
        homography_reproj_rms_px=0.5,
        camera_height_above_table_m=0.33,
        image_size=(640, 480),
        cam_extrinsics_survey1={
            "R_cam_in_base": R_cam_in_base.tolist(),
            "C_cam_in_base_m": C_cam_in_base.tolist(),
            "reproj_rms_px": 0.1,
        },
    )
    # Sanity: same pixel at different fruit heights must produce same XY
    # for a nadir camera, but DIFFERENT XY for this tilted camera. We only
    # check that the result is finite and distinct between the two.
    xyz_low  = pixel_to_base_frame((320, 240), 10.0, cal)
    xyz_high = pixel_to_base_frame((320, 240), 80.0, cal)
    assert np.all(np.isfinite(xyz_low))
    assert np.all(np.isfinite(xyz_high))
    assert not np.allclose(xyz_low[:2], xyz_high[:2])


if __name__ == "__main__":
    _section("B1 Detection fields", test_detection_fields)
    _section("B1 Detection to_dict", test_detection_to_dict)
    _section("B2 hsv_mask single (yellow)", test_hsv_mask_single_range_yellow)
    _section("B2 hsv_mask wrap (red)", test_hsv_mask_red_wrap_around)
    _section("B2 hsv_mask excludes blue", test_hsv_mask_excludes_out_of_range)
    _section("B3 banana finds one", test_banana_contours_finds_one)
    _section("B3 banana rejects round", test_banana_rejects_round_shape)
    _section("B3 banana rejects tiny", test_banana_rejects_tiny_blob)
    _section("B4 tomato finds round red", test_tomato_finds_round_red)
    _section("B4 tomato rejects green-above", test_tomato_rejects_red_with_green_above)
    _section("B4 tomato rejects elongated", test_tomato_rejects_elongated_red)
    _section("B5 strawberry tapered with calyx", test_strawberry_finds_tapered_red_with_calyx)
    _section("B5 strawberry rejects no-calyx", test_strawberry_rejects_red_no_calyx)
    _section("B6 depth sample median", test_depth_sample_returns_median)
    _section("B6 depth sample rejects sparse", test_depth_sample_rejects_mostly_zero)
    _section("B6 depth sample rejects range", test_depth_sample_rejects_out_of_range)
    _section("B7 px->base principal point", test_pixel_to_base_identity_at_principal_point)
    _section("B7 px->base parallax", test_pixel_to_base_parallax_shrinks_with_fruit_height)
    _section("B7 px->base z", test_pixel_to_base_z_equals_origin_plus_fruit_top)
    _section("B8 detect end-to-end", test_detect_fruits_end_to_end)
    _section("B8 detect falls back on null depth", test_detect_fruits_falls_back_on_invalid_depth)
    _section("B8 detect empty scene", test_detect_fruits_empty_scene)
    fails = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{len(_RESULTS)} test(s), {fails} failed")
    sys.exit(fails)
