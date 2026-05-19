"""Unit tests for ugreen_tracker. No hardware needed — tests use synthetic
frames stored as fixtures."""
import os
import sys
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ugreen_tracker import (tcp_from_diff, save_baseline, load_baseline,
                             MIN_ARM_PIXELS)


def _synth_baseline(h=480, w=640):
    """Empty workspace: grey desk with a white paper in the middle."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (440, 340), 240, -1)
    return img


def _synth_with_arm(baseline):
    """Same scene with a dark arm silhouette entering from top-center."""
    img = baseline.copy()
    # Thin vertical arm (dark grey) and a gripper blob at its tip
    cv2.rectangle(img, (310, 0), (330, 260), 40, -1)
    cv2.circle(img, (320, 265), 15, 30, -1)
    return img


def test_diff_finds_arm_tip():
    baseline = _synth_baseline()
    frame = _synth_with_arm(baseline)
    tcp = tcp_from_diff(frame, baseline)
    assert tcp is not None, "expected a TCP, got None"
    col, row = tcp
    # Arm is at column ~320, tip row ~280. Allow loose tolerance.
    assert 300 < col < 340, f"col {col} not near 320"
    assert 260 < row < 295, f"row {row} not near 280"


def test_no_arm_returns_none():
    baseline = _synth_baseline()
    frame = baseline.copy()  # identical; no arm
    tcp = tcp_from_diff(frame, baseline)
    assert tcp is None, f"expected None for identical frames, got {tcp}"


def test_baseline_roundtrip(tmp_path=None):
    import tempfile
    tmp = tempfile.mkdtemp()
    base = _synth_baseline()
    path = os.path.join(tmp, "base.png")
    save_baseline(base, path)
    reloaded = load_baseline(path)
    assert reloaded.shape == base.shape
    assert np.allclose(reloaded, base)


def test_small_difference_rejected():
    """Single-pixel noise shouldn't register as 'arm present'."""
    baseline = _synth_baseline()
    frame = baseline.copy()
    frame[100, 100] = (0, 0, 0)  # one pixel changed
    tcp = tcp_from_diff(frame, baseline)
    assert tcp is None, f"expected None for 1-pixel change, got {tcp}"


def test_tcp_picks_largest_top_touching_component():
    """When multiple silhouettes touch the top edge, pick the largest
    (= the arm, not a shadow strip)."""
    baseline = _synth_baseline()
    frame = baseline.copy()
    # Small shadow strip at top-left — touches y=0, small area
    cv2.rectangle(frame, (50, 0), (70, 40), 40, -1)
    # Full arm at center — touches y=0, much larger area
    cv2.rectangle(frame, (310, 0), (330, 260), 40, -1)
    cv2.circle(frame, (320, 265), 15, 30, -1)
    tcp = tcp_from_diff(frame, baseline)
    assert tcp is not None
    col, _ = tcp
    assert 300 < col < 340, f"picked shadow instead of arm: col={col}"


def test_rejects_whole_frame_diff():
    """Global lighting shift must not be treated as 'arm fills frame'.
    Cap the allowed mask fraction to guard auto-white-balance drift."""
    baseline = _synth_baseline()
    # Simulate a whole-frame brightness shift of 40 gray levels
    frame = np.clip(baseline.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    tcp = tcp_from_diff(frame, baseline)
    assert tcp is None, f"should reject full-frame diff, got {tcp}"


def test_intrinsics_io_roundtrip():
    """Save/load for UGreen intrinsics + distortion."""
    import tempfile, json
    from ugreen_intrinsics import save_intrinsics, load_intrinsics
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "intr.json")
    intr = {'fx': 600.0, 'fy': 601.0, 'cx': 320.0, 'cy': 240.0}
    dist = np.array([0.1, -0.2, 0.001, 0.002, 0.05], dtype=np.float64)
    save_intrinsics(path, intr, dist)
    intr2, dist2 = load_intrinsics(path)
    for k in intr:
        assert abs(intr[k] - intr2[k]) < 1e-9
    assert np.allclose(dist, dist2)


if __name__ == "__main__":
    fails = 0
    for t in [test_diff_finds_arm_tip, test_no_arm_returns_none,
              test_baseline_roundtrip, test_small_difference_rejected,
              test_tcp_picks_largest_top_touching_component,
              test_rejects_whole_frame_diff,
              test_intrinsics_io_roundtrip]:
        try:
            t()
            print(f"  [OK]   {t.__name__}")
        except Exception as ex:
            print(f"  [FAIL] {t.__name__}: {ex}")
            fails += 1
    sys.exit(fails)
