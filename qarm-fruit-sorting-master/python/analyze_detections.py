"""
Zero-risk detection diagnostic.

From pickhome1: grab N frames, run the detector on each, report:
  - per-frame count, per-fruit classification
  - per-fruit centroid stability across frames (std in pixels & mm-base)
  - circularity / saturation of every red blob (diagnose tomato vs strawberry
    misclassifications)

Does not move the arm beyond pickhome1 and does not pick.
"""

import json
import os
import sys
import time
import numpy as np
import cv2

from camera import QArmCamera
from qarm_driver import QArmDriver
import fruit_detector as fd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS_FILE = os.path.join(REPO_ROOT, "teach_points.json")
CALIB_FILE = os.path.join(REPO_ROOT, "calibration.json")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

HOME_LABEL = "pickhome1"
N_FRAMES = 5


def slow_move_to(q, target_joints, target_grip, steps=200, dt=0.02):
    cur_joints, cur_grip = q.read_all()
    cur_joints = np.array(cur_joints, dtype=float)
    cur_grip = float(cur_grip)
    for i in range(1, steps + 1):
        a = i / steps
        s = 3 * a * a - 2 * a * a * a
        q.set_joints_and_gripper(
            cur_joints + s * (target_joints - cur_joints),
            cur_grip + s * (target_grip - cur_grip),
        )
        time.sleep(dt)
    for _ in range(30):
        q.set_joints_and_gripper(target_joints, target_grip)
        time.sleep(0.05)


def classify_red_blobs(bgr):
    """Replicate fruit_detector._extract_red_blobs but return richer stats
    per blob: area, circularity, mean saturation, final classification."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, fd.HSV_RANGES['red']['lower1'],
                     fd.HSV_RANGES['red']['upper1'])
    m2 = cv2.inRange(hsv, fd.HSV_RANGES['red']['lower2'],
                     fd.HSV_RANGES['red']['upper2'])
    mask = fd._clean_mask(cv2.bitwise_or(m1, m2), fd.MIN_AREA)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    stats = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < fd.MIN_AREA:
            continue
        p = cv2.arcLength(c, True)
        if p == 0:
            continue
        circ = 4 * np.pi * area / (p * p)
        bm = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(bm, [c], -1, 255, -1)
        sat = cv2.mean(hsv[:, :, 1], mask=bm)[0]
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        ftype = ('tomato'
                 if circ > fd.CIRCULARITY_THRESH and sat > fd.SATURATION_THRESH
                 else 'strawberry')
        stats.append({'pix': (cx, cy), 'area': area,
                      'circularity': circ, 'saturation': sat,
                      'type': ftype})
    return stats


def patch_depth(depth, row, col, patch=5):
    h, w = depth.shape
    if 0 <= row < h and 0 <= col < w and depth[row, col] > 0:
        return float(depth[row, col])
    r0, r1 = max(0, row - patch), min(h, row + patch + 1)
    c0, c1 = max(0, col - patch), min(w, col + patch + 1)
    win = depth[r0:r1, c0:c1]
    valid = win[win > 0]
    return float(np.median(valid)) if valid.size else None


def main():
    with open(POINTS_FILE) as f:
        pts = json.load(f)
    with open(CALIB_FILE) as f:
        cal = json.load(f)
    T = np.array(cal["T_cam_to_base"], dtype=float)
    print(f"Using calibration: {cal.get('source', 'unknown')}  "
          f"RMS={cal.get('rms_residual_mm', '?')}mm")

    q = QArmDriver()
    q.connect()
    time.sleep(0.5)
    for _ in range(5):
        try:
            q.read_all(); break
        except Exception:
            time.sleep(0.5)
    try:
        if HOME_LABEL in pts:
            target = np.array(pts[HOME_LABEL]["joints_rad"], dtype=float)
            grip = float(pts[HOME_LABEL].get("gripper", 0.15))
            print(f"Moving to {HOME_LABEL} ...")
            slow_move_to(q, target, grip)

        cam = QArmCamera()
        cam.open()
        try:
            # warmup
            deadline = time.time() + 10.0
            while time.time() < deadline:
                try:
                    c, d = cam.read()
                    if c.mean() > 5 and (d > 0).mean() > 0.1:
                        break
                except Exception:
                    pass
                time.sleep(0.1)
            for _ in range(10):
                try:
                    cam.read()
                except Exception:
                    pass
                time.sleep(0.05)

            all_stats = []  # list of (frame_idx, stats_list, depth)
            for k in range(N_FRAMES):
                try:
                    color, depth = cam.read()
                except Exception as ex:
                    print(f"  [frame {k}] read failed: {ex}")
                    continue
                s = classify_red_blobs(color)
                all_stats.append((k, s, depth))
                print(f"\nframe {k}: {len(s)} red blobs")
                for b in sorted(s, key=lambda x: -x['area']):
                    print(f"  pix={b['pix']}  area={b['area']:5.0f}  "
                          f"circ={b['circularity']:.3f}  "
                          f"sat={b['saturation']:5.1f}  -> {b['type']}")
                time.sleep(0.2)

            # Associate blobs across frames by nearest-pixel. Use frame 0 as
            # anchor; for each anchor blob, find the closest blob in each
            # other frame (within 40 px). Report per-object stats.
            if not all_stats:
                print("no frames captured")
                return
            _, anchors, _ = all_stats[0]
            print(f"\n=== Per-object stability ({len(anchors)} anchor blobs, "
                  f"{len(all_stats)} frames) ===")
            for ai, a in enumerate(anchors):
                track = [a]
                for k, s, _ in all_stats[1:]:
                    best = None
                    best_d = 40.0
                    ax, ay = a['pix']
                    for b in s:
                        bx, by = b['pix']
                        dist = np.hypot(bx - ax, by - ay)
                        if dist < best_d:
                            best_d = dist
                            best = b
                    if best is not None:
                        track.append(best)
                if len(track) < 2:
                    print(f"  [obj {ai}] not tracked")
                    continue
                pix = np.array([t['pix'] for t in track], dtype=float)
                circs = np.array([t['circularity'] for t in track])
                sats = np.array([t['saturation'] for t in track])
                types = [t['type'] for t in track]
                # base-frame std via projection
                base_pts = []
                for t, (_, _, depth_k) in zip(track, all_stats[:len(track)]):
                    u, v = t['pix']
                    dmm = patch_depth(depth_k, v, u)
                    if dmm is None:
                        continue
                    p = cam.pixel_to_world(v, u, dmm, T)
                    base_pts.append(p)
                base_arr = (np.array(base_pts)
                            if base_pts else np.zeros((0, 3)))
                print(f"\n  [obj {ai}] anchor pix={a['pix']}  "
                      f"tracked in {len(track)}/{len(all_stats)} frames")
                print(f"    pixel std  : ({pix.std(axis=0)[0]:.2f}, "
                      f"{pix.std(axis=0)[1]:.2f}) px")
                print(f"    circ  range: {circs.min():.3f} .. "
                      f"{circs.max():.3f}  (thresh "
                      f"{fd.CIRCULARITY_THRESH})")
                print(f"    sat   range: {sats.min():.1f} .. {sats.max():.1f}"
                      f"  (thresh {fd.SATURATION_THRESH})")
                flip = len(set(types)) > 1
                print(f"    class: {types}" + ("  [UNSTABLE]" if flip else ""))
                if base_arr.shape[0] >= 2:
                    print(f"    base xyz std (mm): "
                          f"({base_arr.std(axis=0)[0]*1000:.1f}, "
                          f"{base_arr.std(axis=0)[1]*1000:.1f}, "
                          f"{base_arr.std(axis=0)[2]*1000:.1f})")

            # Save the last annotated frame for visual inspection
            dets = fd.detect_fruits(color, depth)
            os.makedirs(LOG_DIR, exist_ok=True)
            out_img = os.path.join(LOG_DIR, "analyze_detections_last.png")
            cv2.imwrite(out_img, fd.draw_detections(color, dets))
            print(f"\nLast annotated frame -> {out_img}")

        finally:
            cam.close()
    finally:
        try:
            q.card.close(); q._connected = False
        except Exception:
            pass


if __name__ == "__main__":
    main()
