"""
Per-pose calibration — captures from a teach-point pose, detects fruits,
brute-forces the 4! / 5040 / n! assignment to ground-truth labels, solves
Umeyama, and writes calibration_<pose_label>.json.

Use this when the camera is arm-mounted and T_cam_to_base changes per pose
(i.e. the single calibration.json is only valid at one fixed pose).

Usage:
    python recal_from_pose.py <pose_label> <gt_label1> <gt_label2> ...
                              [--save] [--min-area N]

Example:
    python recal_from_pose.py placehome1 cal_05 cal_06 cal_07 cal_08 --save
"""

import json
import os
import sys
import time
import itertools
import numpy as np
import cv2

from camera import QArmCamera
from qarm_driver import QArmDriver
from fruit_detector import detect_fruits, draw_detections, detection_depth_mm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS_FILE = os.path.join(REPO_ROOT, "teach_points.json")
LOG_DIR = os.path.join(REPO_ROOT, "logs")


def slow_move_to(q, target_joints, target_grip, seconds=3.0, steps=200):
    cj, cg = q.read_all()
    cj = np.array(cj, dtype=float)
    cg = float(cg)
    for i in range(1, steps + 1):
        a = i / steps
        s = 3 * a * a - 2 * a * a * a
        q.set_joints_and_gripper(
            cj + s * (target_joints - cj),
            cg + s * (target_grip - cg),
        )
        time.sleep(seconds / steps)
    for _ in range(20):
        q.set_joints_and_gripper(target_joints, target_grip)
        time.sleep(0.05)


def umeyama(src, dst):
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d
    H = S.T @ D / len(src)
    U, _, Vt = np.linalg.svd(H)
    W = np.eye(3)
    W[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ W @ U.T
    t = mu_d - R @ mu_s
    return R, t


def residuals(src, dst, R, t):
    pred = (R @ src.T).T + t
    return np.linalg.norm(pred - dst, axis=1)


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print(__doc__)
        sys.exit(1)
    pose_label = args[0]
    gt_labels = [a for a in args[1:] if not a.startswith("--")]
    save = "--save" in args
    min_area = 300
    for i, a in enumerate(args):
        if a == "--min-area" and i + 1 < len(args):
            min_area = int(args[i + 1])

    if len(gt_labels) < 3:
        print("Need >= 3 GT labels. Got:", gt_labels)
        sys.exit(1)

    with open(POINTS_FILE) as f:
        pts = json.load(f)

    if pose_label not in pts:
        print(f"ERROR: pose label {pose_label} not in teach_points")
        sys.exit(1)

    missing = [k for k in gt_labels if k not in pts]
    if missing:
        print(f"ERROR: GT labels missing: {missing}")
        sys.exit(1)

    gt_base = np.array([pts[k]["xyz_m"] for k in gt_labels], dtype=float)
    print(f"Pose: {pose_label}")
    print(f"GT labels ({len(gt_labels)}):")
    for lbl, g in zip(gt_labels, gt_base):
        print(f"  {lbl}: {g.round(3)}")

    q = QArmDriver()
    q.connect()
    time.sleep(0.5)
    for _ in range(5):
        try:
            q.read_all(); break
        except Exception:
            time.sleep(0.5)

    try:
        target = np.array(pts[pose_label]["joints_rad"], dtype=float)
        grip = float(pts[pose_label].get("gripper", 0.15))
        print(f"\nMoving to {pose_label} ...")
        slow_move_to(q, target, grip)

        cam = QArmCamera()
        cam.open()
        try:
            # warmup for non-black color + non-empty depth
            deadline = time.time() + 10
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

            color, depth = cam.read()
            print(f"Frame: color mean={color.mean():.1f}, "
                  f"depth valid={(depth>0).mean()*100:.1f}%")

            dets = detect_fruits(color, depth, min_area=min_area)
            # Only red detections (strawberry/tomato) — calibration uses
            # whatever is at the cal_* positions
            red = [d for d in dets if d.fruit_type in ("strawberry","tomato")]
            print(f"Detections: {len(red)} red "
                  f"({sum(d.fruit_type=='strawberry' for d in red)} strawberry "
                  f"+ {sum(d.fruit_type=='tomato' for d in red)} tomato)")

            out_img = os.path.join(LOG_DIR,
                                    f"recal_{pose_label}_scene.png")
            os.makedirs(LOG_DIR, exist_ok=True)
            cv2.imwrite(out_img, draw_detections(color, dets))
            print(f"scene -> {out_img}")

            # Camera-frame points
            intr = cam.intrinsics
            cam_pts = []
            det_info = []
            for d in red:
                dmm = detection_depth_mm(d, depth)
                if dmm is None:
                    print(f"  [skip] blob pix=({d.centroid[1]},{d.centroid[0]})"
                          " no plausible depth")
                    continue
                row, col = d.centroid
                Z = dmm / 1000.0
                X = (col - intr["cx"]) * Z / intr["fx"]
                Y = (row - intr["cy"]) * Z / intr["fy"]
                cam_pts.append([X, Y, Z])
                det_info.append({
                    "type": d.fruit_type,
                    "pix": (col, row),
                    "depth_mm": dmm,
                    "area": d.area,
                })
            cam_pts = np.array(cam_pts, dtype=float)
            n_det = len(cam_pts)
            n_gt = len(gt_labels)
            print(f"usable detections: {n_det}  /  ground-truth: {n_gt}")

            if n_det < 3:
                print("ERROR: need at least 3 detections with valid depth")
                sys.exit(1)

            # --- Brute-force assignment ---
            # Try every permutation of length n_gt on n_det detections.
            # If n_det > n_gt, we pick the best n_gt-subset.
            # If n_det < n_gt, we can only pair n_det points — drop extras.
            if n_det != n_gt:
                print(f"[warn] {n_det} detections vs {n_gt} GT labels — "
                      f"will pair whichever gives best fit")

            k = min(n_det, n_gt)
            from itertools import permutations, combinations
            best_res = float("inf")
            best = None
            # Choose which detections to use
            det_idx_sets = list(combinations(range(n_det), k))
            gt_idx_sets  = list(combinations(range(n_gt), k))
            perm_count = 0
            for d_set in det_idx_sets:
                for g_set in gt_idx_sets:
                    for perm in permutations(g_set):
                        perm_count += 1
                        src = cam_pts[list(d_set)]
                        dst = gt_base[list(perm)]
                        R, t = umeyama(src, dst)
                        res = residuals(src, dst, R, t)
                        rms = float(np.sqrt((res**2).mean()))
                        if rms < best_res:
                            best_res = rms
                            best = (d_set, perm, R, t, res)
            print(f"brute-force tried {perm_count} permutations")

            if best is None:
                print("ERROR: no valid assignment found")
                sys.exit(1)

            d_set, perm, R, t, res = best
            T_new = np.eye(4)
            T_new[:3, :3] = R
            T_new[:3, 3] = t
            print(f"\n=== Best assignment ===")
            for di, gi in zip(d_set, perm):
                lbl = gt_labels[gi]
                info = det_info[di]
                print(f"  {lbl:8s}  <-  det[{di}] {info['type']:10s} "
                      f"pix={info['pix']}  depth={info['depth_mm']:.0f}mm  "
                      f"residual={np.linalg.norm((R @ cam_pts[di]) + t - gt_base[gi])*1000:.1f}mm")
            print(f"\nmean residual : {res.mean()*1000:.1f} mm")
            print(f"max  residual : {res.max()*1000:.1f} mm")
            print(f"RMS  residual : {np.sqrt((res**2).mean())*1000:.1f} mm")
            print(f"\nT_cam_to_base @ {pose_label}:")
            for row in T_new:
                print("  [" + ", ".join(f"{v:+.4f}" for v in row) + "]")

            if save:
                out = {
                    "timestamp": __import__("datetime").datetime.now()
                        .isoformat(timespec="seconds"),
                    "source": f"recal_from_pose.py at {pose_label}",
                    "pose_label": pose_label,
                    "intrinsics": cam.intrinsics,
                    "n_points": int(k),
                    "labels": [gt_labels[gi] for gi in perm],
                    "residuals_mm": [float(r*1000) for r in res],
                    "mean_residual_mm": float(res.mean()*1000),
                    "max_residual_mm": float(res.max()*1000),
                    "rms_residual_mm": float(np.sqrt((res**2).mean())*1000),
                    "T_cam_to_base": T_new.tolist(),
                }
                out_path = os.path.join(REPO_ROOT,
                                         f"calibration_{pose_label}.json")
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"\n[save] wrote {out_path}")
            else:
                print("\nNOT written — pass --save to write "
                      f"calibration_{pose_label}.json")

        finally:
            cam.close()
    finally:
        try:
            q.card.close(); q._connected = False
        except Exception:
            pass


if __name__ == "__main__":
    main()
