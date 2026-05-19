"""
Fruit detector for the overhead-vision pipeline.

See docs/superpowers/specs/2026-04-22-overhead-vision-pipeline-design.md §3.2.

Consumes a D415 RGB frame, a D415 depth frame, and a SessionCal (from D1).
Returns a list of `Detection` objects, each carrying the fruit type,
pixel + base-frame coordinates, and confidence.

Pure OpenCV 4.x; no hardware imports, no ML.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import cv2


@dataclass
class Detection:
    fruit_type: str              # "banana" | "tomato" | "strawberry"
    center_px: tuple             # (cx, cy) in image pixels
    center_base_m: np.ndarray    # shape (3,) XYZ in robot base frame, metres
    confidence: float            # 0..1
    area_px: int
    bbox: tuple                  # (x, y, w, h) — opencv convention
    # XY unit vector in base frame pointing from the red centroid toward
    # the calyx centroid. Strawberry-only and only when green pixels were
    # actually found; banana/tomato leave this as None. Consumed by
    # picker_viewer._pick_one to bias the pick toward the wide body.
    calyx_dir_base_unit: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fruit_type": self.fruit_type,
            "center_px": list(self.center_px),
            "center_base_m": np.asarray(self.center_base_m).tolist(),
            "confidence": float(self.confidence),
            "area_px": int(self.area_px),
            "bbox": list(self.bbox),
            "calyx_dir_base_unit": (None if self.calyx_dir_base_unit is None
                                    else np.asarray(
                                        self.calyx_dir_base_unit).tolist()),
        }


# ------------------------------------------------------------------------
# Default HSV ranges (OpenCV convention: H=0-179, S=0-255, V=0-255).
# Starting values for lab tuning; hsv_tuner.py (D4-AM) can override.
# ------------------------------------------------------------------------
HSV_RANGES = {
    # Widened on 2026-04-22 after lab sampling: banana under warm lighting
    # sits at H≈15 (orange-yellow), and tomato shadows go down to V≈59.
    "banana":      {"h": [14, 40],  "s": [80, 255], "v": [60, 255]},
    "tomato":      {"h_wrap1": [0, 10],  "h_wrap2": [170, 180],
                    "s": [80, 255], "v": [40, 255]},
    "strawberry":  {"h_wrap1": [0, 10],  "h_wrap2": [170, 180],
                    "s": [80, 255], "v": [40, 255]},
    "green_calyx": {"h": [15, 90],  "s": [30, 255], "v": [15, 255]},  # widened 2026-04-27: prior s>=60 missed dim leaves entirely (every diag contour reported 0.0); prior h>=20 missed yellowed/dying calyx leaves whose hue falls into the orange-yellow range (H~15-20). H=15 still excludes red (H<10) so cross-class HSV stays clean.
}

_OPEN_KERNEL = np.ones((5, 5), np.uint8)
_CLOSE_KERNEL = np.ones((9, 9), np.uint8)


def hsv_mask(bgr: np.ndarray, ranges: dict) -> np.ndarray:
    """
    Build a cleaned binary mask from a BGR image using an HSV ranges dict.

    Accepts two shapes:
      single range:    {"h": [lo, hi], "s": [lo, hi], "v": [lo, hi]}
      wrap-around hue: {"h_wrap1": [..], "h_wrap2": [..], "s": [..], "v": [..]}

    Applies MORPH_OPEN 5x5 then MORPH_CLOSE 9x9 to suppress speckle and
    fill small holes.

    Returns uint8 mask of the same HxW as the input; 0 or 255.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_lo, s_hi = ranges["s"]
    v_lo, v_hi = ranges["v"]
    if "h_wrap1" in ranges and "h_wrap2" in ranges:
        lo1 = np.array([ranges["h_wrap1"][0], s_lo, v_lo], dtype=np.uint8)
        hi1 = np.array([ranges["h_wrap1"][1], s_hi, v_hi], dtype=np.uint8)
        lo2 = np.array([ranges["h_wrap2"][0], s_lo, v_lo], dtype=np.uint8)
        hi2 = np.array([ranges["h_wrap2"][1], s_hi, v_hi], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                               cv2.inRange(hsv, lo2, hi2))
    else:
        lo = np.array([ranges["h"][0], s_lo, v_lo], dtype=np.uint8)
        hi = np.array([ranges["h"][1], s_hi, v_hi], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return mask


# ------------------------------------------------------------------------
# Per-class detection. Each _detect_* returns a list of raw hits:
#   [((cx, cy), area_px, (bx, by, bw, bh), confidence), ...]
# The top-level detect_fruits() wraps these into Detection objects with
# base-frame coordinates.
# ------------------------------------------------------------------------
_BANANA_MIN_AREA = 5000     # raised 2026-04-27 from 800: yellowed/dim strawberry calyx leaves fall into banana HSV (H 14-40, S 80+, V 60+) and have leaf-elongated shape that passes the aspect gate; they show up as ~1-3k px ghost bananas sitting on the strawberry's head. Real bananas at survey1 distance are 20k-50k px so 5000 is a safe floor with 4x margin.
_BANANA_MAX_AREA = 80000   # widened 2026-04-22 for survey1 ~33 cm: close-up bananas reach ~50k px
_BANANA_MIN_ASPECT = 1.5   # loosened 2026-04-22: overhead view of a curved banana has aspect ~1.5-2.0; orange-yellow non-banana items are rare on the lab table


def _detect_banana_contours(bgr: np.ndarray) -> list:
    """Find banana blobs (yellow + elongated). Pure 2D; no depth, no base
    coords. Each hit: ((cx, cy), area_px, (x, y, w, h), confidence)."""
    mask = hsv_mask(bgr, HSV_RANGES["banana"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    hits = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < _BANANA_MIN_AREA or area > _BANANA_MAX_AREA:
            continue
        (cx, cy), (w_r, h_r), _ = cv2.minAreaRect(c)
        if w_r < 1 or h_r < 1:
            continue
        aspect = max(w_r, h_r) / min(w_r, h_r)
        if aspect < _BANANA_MIN_ASPECT:
            continue
        x, y, w_b, h_b = cv2.boundingRect(c)
        # Score: 0.5 at MIN aspect (so anything that already passes the
        # aspect gate clears CONFIDENCE_MIN=0.35 cleanly), rising to 1.0
        # at aspect = MIN + 1.0. Rejected contours (aspect < MIN) never
        # reach this line because of the `continue` above.
        aspect_score = 0.5 + min(0.5, (aspect - _BANANA_MIN_ASPECT) / 2.0)
        hits.append(((int(cx), int(cy)), int(area),
                     (int(x), int(y), int(w_b), int(h_b)),
                     float(aspect_score), None))
    return hits


_TOMATO_MIN_AREA = 1500        # raised 2026-04-22: speckle in shadow regions trips 400, real tomatoes are 10k+ px at survey1 distance
_TOMATO_MAX_AREA = 60000       # raised 2026-04-23: closer-to-base chessboard pushes tomato to 45k px; cap was too tight
_TOMATO_MIN_CIRCULARITY = 0.4   # loosened 2026-04-22: specular highlights bite chunks out of tomato contour, dropping effective circ to ~0.50; green-above rejection keeps strawberries separate
_TOMATO_MAX_ASPECT = 1.8        # added 2026-04-22: circ alone can't distinguish shadowed round tomato (circ 0.50) from elongated red chili (circ 0.42). Aspect cap does the separation cleanly — real tomatoes sit at ~1.0-1.7, elongated reds are >2.
_GREEN_ABOVE_BAND_PX = (10, 20)   # band (min, max) px above blob top
_GREEN_ABOVE_RATIO = 0.05          # green pixels / band area threshold


def _has_green_above(bgr: np.ndarray, bbox: tuple) -> float:
    """Return the fraction of green pixels in a band 10-20 px above bbox top.
    0 if bbox hits the image top."""
    x, y, w, h = bbox
    y_lo = max(0, y - _GREEN_ABOVE_BAND_PX[1])
    y_hi = max(0, y - _GREEN_ABOVE_BAND_PX[0])
    if y_hi <= y_lo:
        return 0.0
    band = bgr[y_lo:y_hi, x:x + w]
    if band.size == 0:
        return 0.0
    green_mask = hsv_mask(band, HSV_RANGES["green_calyx"])
    return float((green_mask > 0).mean())


def _has_green_in_top_strip(bgr: np.ndarray, bbox: tuple,
                              strip_pct: float = 0.20) -> float:
    """Fraction of green pixels in the top strip_pct of the bbox INTERIOR.
    Complement to _has_green_above: in overhead views the strawberry
    calyx sits ON TOP of the red body (inside the bbox), not in a band
    above it. Tomato path uses _has_green_above only (to reject
    strawberries appearing as tomatoes). Strawberry path uses
    max(_has_green_above, _has_green_in_top_strip) so both viewing
    angles confirm the calyx."""
    x, y, w, h = bbox
    strip_h = max(1, int(h * strip_pct))
    strip = bgr[y:y + strip_h, x:x + w]
    if strip.size == 0:
        return 0.0
    green_mask = hsv_mask(strip, HSV_RANGES["green_calyx"])
    return float((green_mask > 0).mean())


def _has_green_anywhere_in_bbox(bgr: np.ndarray, bbox: tuple) -> float:
    """Fraction of green pixels anywhere inside the bbox.

    Overhead camera + free-orientation strawberries means the calyx
    can be on any side (including the bottom). Top-strip / above-band
    checks both miss those. This catches them, at the cost of being
    susceptible to background leak into the axis-aligned bbox corners
    — which is acceptable because real tomatoes have no green
    anywhere near them in this workspace."""
    x, y, w, h = bbox
    region = bgr[y:y + h, x:x + w]
    if region.size == 0:
        return 0.0
    green_mask = hsv_mask(region, HSV_RANGES["green_calyx"])
    return float((green_mask > 0).mean())


def _calyx_signal(bgr: np.ndarray, bbox: tuple) -> float:
    """Strongest evidence of a green calyx anywhere in / above the bbox.

    Used by BOTH the strawberry positive gate and the tomato
    rejection gate so the two classes stay mutually exclusive on
    the green criterion (otherwise a contour with a small calyx
    visible only in top_strip could pass tomato AND strawberry,
    producing two Detection records on the same blob)."""
    return max(
        _has_green_above(bgr, bbox),
        _has_green_in_top_strip(bgr, bbox),
        _has_green_anywhere_in_bbox(bgr, bbox),
    )


def _detect_tomato_contours(bgr: np.ndarray) -> list:
    mask = hsv_mask(bgr, HSV_RANGES["tomato"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    hits = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < _TOMATO_MIN_AREA or area > _TOMATO_MAX_AREA:
            continue
        perim = float(cv2.arcLength(c, True))
        if perim < 1:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < _TOMATO_MIN_CIRCULARITY:
            continue
        x, y, w_b, h_b = cv2.boundingRect(c)
        # Use axis-aligned bbox aspect (not minAreaRect): specular
        # highlights bite into the contour and make minAreaRect pick a
        # rotated frame with inflated aspect. The axis-aligned bbox
        # reflects "roundness" in the image plane, which is what we want.
        if min(w_b, h_b) > 0:
            aspect = max(w_b, h_b) / min(w_b, h_b)
            if aspect > _TOMATO_MAX_ASPECT:
                continue
        green_ratio = _calyx_signal(bgr, (x, y, w_b, h_b))
        if green_ratio > _GREEN_ABOVE_RATIO:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        conf = float(min(1.0, circ) * (1.0 - green_ratio))
        hits.append(((cx, cy), int(area),
                     (int(x), int(y), int(w_b), int(h_b)), conf, None))
    return hits


_STRAWBERRY_MIN_AREA = 1500       # raised 2026-04-27 from 200: gripper edges and red specks at frame edges (areas 200-600 px) were passing the new shape signal (circ < 0.4) and showing up as ghost strawberries. Real lab strawberries at survey1 distance are 20k+ px, so 1500 (= _TOMATO_MIN_AREA) gives a 14x safety margin.
_STRAWBERRY_MAX_AREA = 40000      # widened 2026-04-22: close-up strawberries reach ~30k px at survey1 distance
_STRAWBERRY_MIN_CALYX = 0.05     # same threshold as tomato's rejection band
_STRAWBERRY_MAX_CIRC = 0.4       # added 2026-04-27: when calyx is not visible (camera-facing-away orientation, brown/dim leaves), an irregular shape (circ < 0.4) is itself a strawberry signal. Set equal to _TOMATO_MIN_CIRCULARITY so the two classes never overlap on the shape axis (tomato accepts circ >= 0.4, strawberry shape-signal fires at circ < 0.4).


def _taper_score(contour, bbox) -> float:
    """Return width_at_top_20% / width_at_bottom_80% of the contour.

    Images use image-coord convention (y grows DOWN), so the "top" band
    corresponds to the blob's upper edge (where the strawberry's calyx
    attaches) and is the WIDER part. A taper > 1.0 means wider top,
    narrower bottom (strawberry shape). ~1.0 means tomato. Our
    _draw_strawberry_bgr test fixture produces taper ≈ 1.9."""
    x, y, w, h = bbox
    if h < 10:
        return 1.0
    top_band_y = y + int(h * 0.2)
    bot_band_y = y + int(h * 0.8)
    mask = np.zeros((y + h + 2, x + w + 2), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    top_cols = np.where(mask[top_band_y] > 0)[0]
    bot_cols = np.where(mask[bot_band_y] > 0)[0]
    top_w = (top_cols.max() - top_cols.min()) if top_cols.size >= 2 else 1
    bot_w = (bot_cols.max() - bot_cols.min()) if bot_cols.size >= 2 else 1
    return float(top_w) / float(max(1, bot_w))


def _widest_point_centroid(contour, bgr_shape) -> tuple | None:
    """Pixel that lies furthest from any contour edge — the centre of the
    largest inscribed disk. For a teardrop strawberry this lands inside
    the WIDE body, not at the moments centroid which gets pulled toward
    the narrow tip. For circles it falls near the geometric centre.
    Returns None if the contour has no interior pixels."""
    H, W = bgr_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    x0 = max(0, x - 1); y0 = max(0, y - 1)
    x1 = min(W, x + w + 1); y1 = min(H, y + h + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, :, 0] -= x0
    shifted[:, :, 1] -= y0
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    if int(mask.sum()) == 0:
        return None
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    yi, xi = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return (x0 + int(xi), y0 + int(yi))


def _detect_strawberry_contours(bgr: np.ndarray) -> list:
    mask = hsv_mask(bgr, HSV_RANGES["strawberry"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    hits = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < _STRAWBERRY_MIN_AREA or area > _STRAWBERRY_MAX_AREA:
            continue
        x, y, w_b, h_b = cv2.boundingRect(c)
        calyx_ratio = _calyx_signal(bgr, (x, y, w_b, h_b))
        # Shape signal: low circularity (irregular shape) is a positive
        # strawberry indicator since tomato is always round (>= 0.4 by
        # _TOMATO_MIN_CIRCULARITY). Used as a fallback when the calyx
        # green is not detectable in this contour's bbox.
        perim = float(cv2.arcLength(c, True))
        if perim < 1:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        shape_signal = circ < _STRAWBERRY_MAX_CIRC
        if calyx_ratio <= _STRAWBERRY_MIN_CALYX and not shape_signal:
            continue
        taper = _taper_score(c, (x, y, w_b, h_b))
        # Widened 2026-04-22: in overhead view (camera nearly nadir) the
        # strawberry's 3-D taper collapses onto a roughly circular blob
        # and the 2-D taper loses diagnostic power. Keep as a sanity gate
        # only: reject extreme elongation in either direction (< 0.3 =
        # pencil-thin at top, > 3.5 = pencil-thin at bottom, both are
        # segmentation artifacts).
        if taper < 0.3 or taper > 3.5:
            continue
        # Use the widest-inscribed-disk centre rather than the mass
        # centroid so the gripper targets the WIDE body of a teardrop
        # strawberry rather than a point biased toward the narrow tip.
        widest = _widest_point_centroid(c, bgr.shape)
        if widest is None:
            continue
        cx, cy = widest
        # Confidence: prefer calyx evidence (continuous), use shape as
        # a mid-confidence floor when only shape says strawberry. Both
        # paths are independently sufficient and CONFIDENCE_MIN is the
        # final gate.
        calyx_conf = float(min(1.0, calyx_ratio * 3.0))
        shape_conf = 0.5 if shape_signal else 0.0
        conf = max(calyx_conf, shape_conf)
        # Calyx pixel centroid for the pick-bias step. Search a padded
        # region around the bbox so above-bbox calyx (the standard
        # upright orientation) is also captured. None if green_calyx
        # mask is empty in the region (shape-only strawberry).
        green_px = _calyx_centroid_px(bgr, (x, y, w_b, h_b))
        hits.append(((cx, cy), int(area),
                     (int(x), int(y), int(w_b), int(h_b)),
                     conf, green_px))
    return hits


def _calyx_centroid_px(bgr: np.ndarray, bbox: tuple,
                         pad: int = 25) -> tuple | None:
    """Return (gcx, gcy) image-space centroid of green pixels in the
    bbox + padding. None if no green pixels found."""
    x, y, w, h = bbox
    y0 = max(0, y - pad)
    y1 = min(bgr.shape[0], y + h + pad)
    x0 = max(0, x - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    region = bgr[y0:y1, x0:x1]
    if region.size == 0:
        return None
    green_mask = hsv_mask(region, HSV_RANGES["green_calyx"])
    rows, cols = np.where(green_mask > 0)
    if rows.size == 0:
        return None
    gcx = float(x0 + cols.mean())
    gcy = float(y0 + rows.mean())
    return (gcx, gcy)


_DEPTH_MIN_VALID = 8       # min valid samples in the patch
_DEPTH_MIN_MM = 10
_DEPTH_MAX_MM = 800


def sample_depth_at_pixel(depth_mm: np.ndarray, center_px: tuple,
                           patch: int = 5,
                           min_mm: int = _DEPTH_MIN_MM,
                           max_mm: int = _DEPTH_MAX_MM):
    """Return median depth (in mm) of a patch around (cx, cy), or None
    if the patch has fewer than _DEPTH_MIN_VALID samples inside
    [min_mm, max_mm]."""
    cx, cy = int(center_px[0]), int(center_px[1])
    half = patch // 2
    y_lo = max(0, cy - half)
    y_hi = min(depth_mm.shape[0], cy + half + 1)
    x_lo = max(0, cx - half)
    x_hi = min(depth_mm.shape[1], cx + half + 1)
    patch_vals = depth_mm[y_lo:y_hi, x_lo:x_hi].astype(np.int32).ravel()
    valid = patch_vals[(patch_vals >= min_mm) & (patch_vals <= max_mm)]
    if valid.size < _DEPTH_MIN_VALID:
        return None
    return float(np.median(valid))


def pixel_to_base_frame(center_px: tuple, fruit_top_z_mm: float,
                         session_cal) -> np.ndarray:
    """Convert (pixel, fruit_top_z_mm) to base-frame XYZ (metres).

    `fruit_top_z_mm` is measured ABOVE THE TABLE SURFACE (not in the
    robot base frame) — i.e. the target plane in base frame is
    `z_base = chess_origin_in_base_m[2] + fruit_top_z_mm / 1000`.
    This convention is shared with `_resolve_top_z` / `_FRUIT_TOP_Z_MM`.

    Uses the per-session camera extrinsics recovered at survey1 by
    cv2.solvePnP (see calibrate_extrinsics.solve_survey1_extrinsics).
    Back-projects the pixel into a camera-frame ray, transforms the ray
    into base frame using session_cal.cam_extrinsics_survey1, and
    intersects it with the target plane above.
    """
    extr = session_cal.cam_extrinsics_survey1
    if not extr:
        raise ValueError(
            "session_cal.cam_extrinsics_survey1 is missing — "
            "rerun calibrate_chessboard.py to solvePnP the survey pose")
    intr = session_cal.d415_intrinsics
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx_p, cy_p = float(intr["cx"]), float(intr["cy"])
    u, v = float(center_px[0]), float(center_px[1])

    ray_cam = np.array([(u - cx_p) / fx, (v - cy_p) / fy, 1.0],
                        dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)

    R_cam_in_base = np.asarray(extr["R_cam_in_base"], dtype=np.float64)
    C = np.asarray(extr["C_cam_in_base_m"], dtype=np.float64)
    ray_base = R_cam_in_base @ ray_cam

    # fruit_top_z_mm is height above table surface; offset into base frame.
    origin_z = float(session_cal.chess_origin_in_base_m[2])
    target_z = origin_z + float(fruit_top_z_mm) / 1000.0
    if abs(ray_base[2]) < 1e-6:
        raise ValueError("ray is parallel to table plane; no intersection")
    lam = (target_z - C[2]) / ray_base[2]
    if lam <= 0:
        raise ValueError("ray-plane intersection behind camera; invalid")
    return C + lam * ray_base


CONFIDENCE_MIN = 0.35

# Per-type fruit top-of-fruit height above the table surface (mm).
# Used when D415 depth is missing OR outside a plausible range. In the
# 2026-04-22 D4 smoke test, survey1 was ~42 deg off nadir and D415 depth
# along the oblique ray gave values ~2x the true vertical height. Falling
# back to these defaults gives stable base-Z for picking.
_FRUIT_TOP_Z_MM = {
    "banana": 25,
    "tomato": 55,     # 2026-04-24: measured ~5.5 cm on the lab tomato
    "strawberry": 25,
}


def _resolve_top_z(fruit_type: str, depth_mm: np.ndarray, center_px: tuple,
                    session_cal) -> float:
    """Return the fruit-top Z in mm (above the chess plane) to use for
    parallax + base-frame projection. Prefers D415 depth when it falls
    inside a plausible band; otherwise uses the per-type default."""
    default_z = float(_FRUIT_TOP_Z_MM.get(fruit_type, 30))
    h_mm = float(session_cal.camera_height_above_table_m) * 1000.0
    # Plausible fruit top heights: 5 mm (crumb) to h_mm - 10 mm (nearly at camera).
    # Anything below this band is noise; above this band is wall/background.
    sampled = sample_depth_at_pixel(depth_mm, center_px)
    if sampled is None:
        return default_z
    # Depth from D415 is (approximately) distance-along-ray, so on a tilted
    # survey pose it can exceed camera_height. Reject anything > h_mm or
    # < 5 mm as garbage and fall back to the default.
    if sampled > h_mm or sampled < 5:
        return default_z
    # Convert depth (distance from camera to fruit top) into fruit-top height
    # above the table: fruit_top_z = h_mm - sampled.
    return max(5.0, h_mm - float(sampled))


def detect_fruits(color_bgr: np.ndarray, depth_mm: np.ndarray,
                   session_cal) -> list[Detection]:
    """
    Top-level detector. Returns every fruit found with confidence
    >= CONFIDENCE_MIN. Uses D415 depth when available+plausible;
    falls back to per-type default heights otherwise (see _resolve_top_z).

    Processing order: banana -> tomato -> strawberry. This order does
    not matter for correctness — each detector runs independently on
    its own HSV mask — but keeps the returned list deterministic.
    """
    results: list[Detection] = []
    per_class = [
        ("banana", _detect_banana_contours),
        ("tomato", _detect_tomato_contours),
        ("strawberry", _detect_strawberry_contours),
    ]
    for fruit_type, fn in per_class:
        for (cx, cy), area, bbox, conf, green_px in fn(color_bgr):
            if conf < CONFIDENCE_MIN:
                continue
            top_z_mm = _resolve_top_z(
                fruit_type, depth_mm, (cx, cy), session_cal)
            try:
                xyz_base = pixel_to_base_frame(
                    (cx, cy), top_z_mm, session_cal)
            except ValueError:
                continue
            calyx_dir = None
            if green_px is not None:
                try:
                    green_xyz = pixel_to_base_frame(
                        green_px, top_z_mm, session_cal)
                    diff = green_xyz[:2] - xyz_base[:2]
                    norm = float(np.linalg.norm(diff))
                    if norm > 1e-4:
                        calyx_dir = diff / norm
                except ValueError:
                    pass
            results.append(Detection(
                fruit_type=fruit_type,
                center_px=(int(cx), int(cy)),
                center_base_m=xyz_base,
                confidence=float(conf),
                area_px=int(area),
                bbox=tuple(int(v) for v in bbox),
                calyx_dir_base_unit=calyx_dir,
            ))
    return results
