"""
Detect the post-it markers on the QArm gripper in a UGreen frame.
Returns the marker centroid in pixel coordinates — the projection of the
gripper TCP into the ground-level camera view. Used for closed-loop
calibration and pick validation.

The UGreen's auto white-balance washes the purple post-its to near-white,
so we threshold on V>THRESH (bright) AND S<SAT_MAX (desaturated) which
catches white / near-white pixels, then restrict to a region of interest
around the expected gripper area to reject ceiling lights / white walls.
"""
import os
import cv2
import numpy as np

# Thresholds: bright AND low saturation = white-ish
V_MIN = 200
S_MAX = 60
MIN_AREA = 150
MAX_AREA = 20000

# Gripper ROI in UGreen frame (x0, y0, x1, y1) — covers the area where the
# arm appears at common poses. Tune this per setup. None = full frame.
DEFAULT_ROI = (420, 10, 820, 260)


def detect_gripper_markers(bgr_image, roi=DEFAULT_ROI, debug=False):
    """Find bright / near-white blobs in the gripper ROI. Returns list of
    dicts with 'pix' (col,row), 'area', 'bbox', 'aspect' sorted by area
    descending. When debug=True also returns the threshold mask."""
    h, w = bgr_image.shape[:2]
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, (0, 0, V_MIN), (179, S_MAX, 255))

    # Zero out everything outside the ROI so distant bright objects
    # (ceiling lights, white shirts, walls) don't register.
    mask = np.zeros_like(bright)
    if roi is not None:
        x0, y0, x1, y1 = roi
        x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
        mask[y0:y1, x0:x1] = bright[y0:y1, x0:x1]
    else:
        mask = bright

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = (max(bw, bh) / max(1, min(bw, bh)))
        out.append({'pix': (cx, cy), 'area': float(area),
                    'bbox': (x, y, bw, bh), 'aspect': float(aspect)})

    out.sort(key=lambda d: -d['area'])

    if debug:
        return out, mask
    return out


# Backward-compat alias
detect_purple_markers = detect_gripper_markers


def draw_markers(bgr, markers):
    """Overlay detected markers with crosshair + area label."""
    vis = bgr.copy()
    for i, m in enumerate(markers):
        cx, cy = m['pix']
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=30, thickness=2)
        x, y, w, h = m['bbox']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, f"#{i} A={m['area']:.0f}",
                    (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)
    return vis


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "logs",
        "ugreen_latest.png")
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"could not read {path}"); sys.exit(1)
    markers, mask = detect_purple_markers(bgr, debug=True)
    print(f"found {len(markers)} purple markers:")
    for i, m in enumerate(markers):
        print(f"  #{i} pix={m['pix']}  area={m['area']:.0f}")
    out_vis = path.replace(".png", "_markers.png")
    out_mask = path.replace(".png", "_mask.png")
    cv2.imwrite(out_vis, draw_markers(bgr, markers))
    cv2.imwrite(out_mask, mask)
    print(f"annotated -> {out_vis}")
    print(f"mask      -> {out_mask}")
