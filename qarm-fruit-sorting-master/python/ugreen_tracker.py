"""UGreen floor-level camera tracker.

Detects the QArm gripper in a UGreen frame via baseline-subtraction image
diff. Returns the pixel coordinates of the arm's TCP approximation (the
bottom-most point of the moving silhouette). Used for closed-loop
calibration and pick validation.

Camera access: DirectShow index 3, MSMF backend, 1280x720. Index/backend
discovered by `python/probe_ugreen.py` on 2026-04-20.
"""
import os
import sys
import cv2
import numpy as np

UGREEN_IDX = 3
DIFF_THRESH = 35            # absolute mean channel diff to count a pixel as "arm"
MIN_ARM_PIXELS = 500        # total arm silhouette must exceed this
MAX_ARM_PIXELS_FRAC = 0.50  # reject masks that cover > 50% of frame
                            # (catches global lighting shifts / auto-WB drift)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Column band where the QArm base mounts — an "arm" component's centroid
# must fall inside this band. Rejects background motion (people, doors)
# that also touches the top edge of the frame but sits off to one side.
# Tuned for the 2026-04-20 UGreen setup (arm front-center, 1280 wide).
ARM_COL_BAND = (380, 820)


def capture(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, warmup=10):
    """Open UGreen, grab one valid frame, release. Returns BGR ndarray.
    Raises RuntimeError if the camera does not yield a usable frame
    within the warmup budget."""
    cap = cv2.VideoCapture(UGREEN_IDX, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"UGreen did not open (idx={UGREEN_IDX} MSMF)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(max(1, warmup)):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0 and f.mean() > 5:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("UGreen opened but no valid frame after warmup")
    return frame


def save_baseline(frame, path):
    """Persist a baseline frame (arm out of view) to disk. Used by
    preflight + closed-loop calibration."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not cv2.imwrite(path, frame):
        raise IOError(f"cv2.imwrite failed for {path}")


def load_baseline(path):
    """Load a previously saved baseline. Raises FileNotFoundError if
    missing — preflight is expected to catch that."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"cv2.imread returned None for {path}")
    return img


def arm_mask(frame, baseline, thresh=DIFF_THRESH, col_band=ARM_COL_BAND):
    """Binary mask where |frame - baseline| exceeds `thresh`. Reduced to
    grayscale by averaging the 3 BGR channels so lighting shifts affect
    all channels equally (the threshold then absorbs small global shifts).

    When `col_band=(lo, hi)` is provided, columns outside [lo, hi] are
    zeroed BEFORE morphological cleanup. This prevents background motion
    (people behind the bench, doors) from merging with the arm silhouette
    via the MORPH_CLOSE operation. Pass `col_band=None` to disable.
    """
    if frame.shape != baseline.shape:
        raise ValueError(f"shape mismatch: {frame.shape} vs {baseline.shape}")
    diff = cv2.absdiff(frame, baseline).mean(axis=2)
    mask = (diff > thresh).astype(np.uint8) * 255
    if col_band is not None:
        lo, hi = col_band
        w = mask.shape[1]
        # Only apply if the frame is wide enough for the band to make sense.
        # Synthetic 640-wide test frames get skipped; the real 1280-wide
        # UGreen frames get the background-rejection benefit.
        if w >= 1000:
            if lo > 0:
                mask[:, :min(lo, w)] = 0
            if hi < w:
                mask[:, max(hi, 0):] = 0
    # Morphological cleanup — small kernel to keep thin arm features
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def tcp_from_diff(frame, baseline, thresh=DIFF_THRESH,
                   min_pixels=MIN_ARM_PIXELS,
                   col_band=ARM_COL_BAND):
    """Locate the gripper TCP as the bottom-most point of the connected
    component reaching from the top of the frame. Returns (col, row) in
    pixel space, or None if no plausible arm silhouette is present.

    Algorithm
    ---------
    1. Compute |frame - baseline| and threshold.
    2. Find connected components. Reject if total arm pixels < min_pixels.
    3. Pick the component whose bounding box touches the top edge
       (y=0), i.e. the arm enters from above. If multiple touch the top,
       pick the largest by area.
    4. Return the bottom-center pixel of that component's bounding box.
    """
    mask = arm_mask(frame, baseline, thresh=thresh, col_band=col_band)
    mask_pixels = int((mask > 0).sum())
    if mask_pixels < min_pixels:
        return None
    if mask_pixels > MAX_ARM_PIXELS_FRAC * int(mask.size):
        return None  # global lighting shift — not a real arm
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_i = -1
    best_area = 0
    best_bbox = None
    for i in range(1, num):  # skip background label 0
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = int(stats[i, cv2.CC_STAT_AREA])
        if y > 10:
            continue  # does not reach the top — not the arm
        if area > best_area:
            best_area = area
            best_i = i
            best_bbox = (x, y, w, h)
    if best_i < 0 or best_area < min_pixels:
        return None
    x, y, w, h = best_bbox
    col = int(x + w / 2)
    row = int(y + h)  # bottom of the bounding box
    return (col, row)


def overlay(frame, baseline, tcp=None):
    """Return a BGR image with the arm mask tinted cyan and the TCP
    marked with a crosshair. Used by preflight visual checks and the
    remote-mode companion view."""
    mask = arm_mask(frame, baseline)
    vis = frame.copy()
    tint = np.zeros_like(vis)
    tint[:] = (255, 255, 0)  # cyan in BGR
    m3 = cv2.merge([mask, mask, mask]).astype(bool)
    vis[m3] = cv2.addWeighted(vis, 0.5, tint, 0.5, 0)[m3]
    if tcp is not None:
        cv2.drawMarker(vis, tcp, (0, 255, 255),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=40, thickness=2)
    return vis


if __name__ == "__main__":
    # Quick manual test against the saved frame from 2026-04-20
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    latest = os.path.join(repo, "logs", "ugreen_latest.png")
    if not os.path.exists(latest):
        print(f"no frame at {latest}; run ugreen_capture first")
        sys.exit(1)
    frame = cv2.imread(latest)
    print(f"loaded {frame.shape}")
    if len(sys.argv) > 1 and sys.argv[1] == "--baseline":
        out = os.path.join(repo, "logs", "ugreen_baseline.png")
        save_baseline(frame, out)
        print(f"saved baseline -> {out}")
        sys.exit(0)
    base_path = os.path.join(repo, "logs", "ugreen_baseline.png")
    if not os.path.exists(base_path):
        print(f"no baseline at {base_path}; save one with --baseline")
        sys.exit(1)
    base = load_baseline(base_path)
    tcp = tcp_from_diff(frame, base)
    print(f"TCP = {tcp}")
    vis = overlay(frame, base, tcp)
    out = os.path.join(repo, "logs", "ugreen_tcp_overlay.png")
    cv2.imwrite(out, vis)
    print(f"overlay -> {out}")
