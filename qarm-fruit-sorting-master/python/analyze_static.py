"""
Static calibration sanity check.

1. Moves the arm to homeplace0 (out of camera view).
2. Opens the D415, runs fruit_detector on one frame.
3. For each detected strawberry, projects (u, v, depth) to base frame
   using calibration.json's T_cam_to_base.
4. Associates each ground-truth teach point (point1..point4) to the
   nearest detection and reports the delta.

Does NOT touch the fruit or attempt to pick. Safe to re-run.
"""

import json
import os
import sys
import time
import numpy as np
import cv2

from camera import QArmCamera
from qarm_driver import QArmDriver
from fruit_detector import detect_fruits, draw_detections, detection_depth_mm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS_FILE = os.path.join(REPO_ROOT, "teach_points.json")
CALIB_FILE = os.path.join(REPO_ROOT, "calibration.json")
OUT_IMG = os.path.join(REPO_ROOT, "logs", "analyze_static_scene.png")

GT_LABELS_DEFAULT = ["point1", "point2", "point3", "point4"]
HOME_LABEL = "pickhome1"


def parse_labels(argv):
    labs = []
    skip = False
    for a in argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--from":
            skip = True
            continue
        if a.startswith("--"):
            continue
        labs.append(a)
    return labs if labs else GT_LABELS_DEFAULT


def parse_home(argv, default):
    for i, a in enumerate(argv[1:]):
        if a == "--from" and i + 1 < len(argv) - 1:
            return argv[i + 2]
    return default
MOVE_STEPS = 200
MOVE_DT = 0.02


def slow_move_to(q, target_joints, target_grip, steps=MOVE_STEPS, dt=MOVE_DT):
    cur_joints, cur_grip = q.read_all()
    cur_joints = np.array(cur_joints, dtype=float)
    cur_grip = float(cur_grip)
    for i in range(1, steps + 1):
        a = i / steps
        s = 3 * a * a - 2 * a * a * a
        phi = cur_joints + s * (target_joints - cur_joints)
        g = cur_grip + s * (target_grip - cur_grip)
        q.set_joints_and_gripper(phi, g)
        time.sleep(dt)
    for _ in range(30):
        q.set_joints_and_gripper(target_joints, target_grip)
        time.sleep(0.05)


# Plausible workspace depth band from camera at pickhome1 (empirical):
# fruits sit ~330-470mm, table up to ~600mm. Anything beyond this is
# background (wall / distant scene) and should be rejected.
DEPTH_MIN_MM = 200
DEPTH_MAX_MM = 700


def patch_depth(depth, row, col, patch=5):
    """Depth at (row,col) in mm, with progressively larger patch fallbacks
    for IR-shadow regions, clamped to plausible workspace range."""
    h, w = depth.shape
    if 0 <= row < h and 0 <= col < w:
        d = int(depth[row, col])
        if DEPTH_MIN_MM <= d <= DEPTH_MAX_MM:
            return float(d)
    for p in (patch, patch * 3, patch * 6):
        r0, r1 = max(0, row - p), min(h, row + p + 1)
        c0, c1 = max(0, col - p), min(w, col + p + 1)
        win = depth[r0:r1, c0:c1]
        valid = win[(win >= DEPTH_MIN_MM) & (win <= DEPTH_MAX_MM)]
        if valid.size > 20:
            return float(np.median(valid))
    return None


def main():
    with open(POINTS_FILE) as f:
        pts = json.load(f)
    with open(CALIB_FILE) as f:
        cal = json.load(f)
    T = np.array(cal["T_cam_to_base"], dtype=float)

    GT_LABELS = parse_labels(sys.argv)
    home_label = parse_home(sys.argv, HOME_LABEL)
    print(f"Capture pose: {home_label}")
    missing = [k for k in GT_LABELS if k not in pts]
    if missing:
        print(f"ERROR: missing teach points: {missing}")
        sys.exit(1)
    gt = {k: np.array(pts[k]["xyz_m"], dtype=float) for k in GT_LABELS}
    print(f"Ground-truth labels: {GT_LABELS}")

    # --- Home the arm out of the camera's view ---
    print(f"Connecting to QArm, moving to {HOME_LABEL} ...")
    q = QArmDriver()
    q.connect()
    # Give the board a moment to settle, then retry the first read
    time.sleep(0.5)
    for attempt in range(5):
        try:
            q.read_all()
            break
        except Exception as ex:
            print(f"  [warn] read_all attempt {attempt + 1} failed: {ex}")
            time.sleep(0.5)
    else:
        print("ERROR: QArm not responding — power-cycle the robot and retry.")
        try:
            q.card.close()
        except Exception:
            pass
        sys.exit(1)
    try:
        if home_label in pts:
            target = np.array(pts[home_label]["joints_rad"], dtype=float)
            grip = float(pts[home_label].get("gripper", 0.15))
            print(f"Moving to {home_label} ...")
            slow_move_to(q, target, grip)
        else:
            print(f"  [warn] {home_label} not in teach_points — skipping move")

        # --- Open camera + warmup ---
        print("Opening camera ...")
        cam = QArmCamera()
        cam.open()
        try:
            # Wait up to 10 s for a non-black color frame AND non-empty depth
            deadline = time.time() + 10.0
            color = np.zeros((720, 1280, 3), dtype=np.uint8)
            depth = np.zeros((720, 1280), dtype=np.uint16)
            while time.time() < deadline:
                try:
                    color, depth = cam.read()
                except Exception:
                    pass
                color_ok = color.mean() > 5
                depth_ok = (depth > 0).mean() > 0.10
                if color_ok and depth_ok:
                    break
                time.sleep(0.1)

            # A few extra frames so auto-exposure has fully settled
            for _ in range(10):
                try:
                    color, depth = cam.read()
                except Exception:
                    pass
                time.sleep(0.05)
            print(f"Frame: color {color.shape} (mean={color.mean():.1f})  "
                  f"depth {depth.shape}  "
                  f"depth_valid={(depth > 0).mean() * 100:.1f}%")

            # --- Detect fruits ---
            dets = detect_fruits(color, depth)
            straws = [d for d in dets if d.fruit_type == "strawberry"]
            toms = [d for d in dets if d.fruit_type == "tomato"]
            bans = [d for d in dets if d.fruit_type == "banana"]
            print(f"Detections: {len(straws)} strawberry, {len(toms)} tomato, "
                  f"{len(bans)} banana")

            os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
            cv2.imwrite(OUT_IMG, draw_detections(color, dets))
            print(f"Scene saved to {OUT_IMG}")

            # Project all red detections (strawberry + tomato) — the detector
            # may classify any of the placed strawberries either way if the
            # circularity threshold is near the boundary.
            candidates = straws + toms
            results = []
            for d in candidates:
                row, col = d.centroid
                d_mm = detection_depth_mm(d, depth)
                if d_mm is None:
                    print(f"  [skip] {d.fruit_type} at pix=({col},{row}) "
                          f"no valid depth in blob")
                    continue
                xyz_base = cam.pixel_to_world(row, col, d_mm, T)
                results.append({
                    "type": d.fruit_type,
                    "pix": (col, row),
                    "depth_mm": d_mm,
                    "xyz_base": xyz_base,
                    "area": d.area,
                })

            print()
            print("=== All red/pink detections (camera -> base via calib) ===")
            for i, r in enumerate(results):
                print(f"  [{i}] {r['type']:10s} pix={r['pix']} "
                      f"depth={r['depth_mm']:.0f}mm  "
                      f"xyz_base={r['xyz_base'].round(3)}  "
                      f"area={r['area']:.0f}")

            # --- Associate each GT point to its detection by PIXEL proximity
            # (project GT through calibration into image, then match on u,v).
            # This is robust to systematic XY base-frame bias in the calib.
            T_base_to_cam = np.linalg.inv(T)
            intr = cam.intrinsics

            def project_gt(p_base):
                p = T_base_to_cam @ np.array([*p_base, 1.0])
                if p[2] <= 0:
                    return None
                u = intr["fx"] * p[0] / p[2] + intr["cx"]
                v = intr["fy"] * p[1] / p[2] + intr["cy"]
                return (u, v)

            print()
            print("=== Ground-truth vs detection (pixel-space matching) ===")
            deltas = []
            used = set()
            for label in GT_LABELS:
                g = gt[label]
                expected_uv = project_gt(g)
                if expected_uv is None:
                    print(f"  {label}: GT projects behind camera, skip")
                    continue
                eu, ev = expected_uv
                best_i = -1
                best_px = float("inf")
                for i, r in enumerate(results):
                    if i in used:
                        continue
                    cu, cv = r["pix"]
                    dpix = float(np.hypot(cu - eu, cv - ev))
                    if dpix < best_px:
                        best_px = dpix
                        best_i = i
                if best_i < 0:
                    print(f"  {label}: no detection available")
                    continue
                used.add(best_i)
                r = results[best_i]
                delta = r["xyz_base"] - g
                deltas.append(delta)
                print(f"  ({label} expected pix=({eu:.0f},{ev:.0f}), "
                      f"matched det pix={r['pix']}, pixel dist={best_px:.0f})")
                print(f"  {label}:")
                print(f"    gt   xyz = [{g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}]")
                print(f"    det  xyz = [{r['xyz_base'][0]:+.3f} "
                      f"{r['xyz_base'][1]:+.3f} {r['xyz_base'][2]:+.3f}]  "
                      f"({r['type']}, pix={r['pix']}, "
                      f"depth={r['depth_mm']:.0f}mm)")
                print(f"    delta    = [{delta[0]:+.3f} "
                      f"{delta[1]:+.3f} {delta[2]:+.3f}]  "
                      f"|d|={np.linalg.norm(delta) * 1000:.0f}mm")

            if deltas:
                D = np.stack(deltas)
                print()
                print("=== Error summary ===")
                print(f"  n paired        : {len(deltas)}")
                print(f"  mean  |delta|   : "
                      f"{np.linalg.norm(D, axis=1).mean() * 1000:.1f} mm")
                print(f"  max   |delta|   : "
                      f"{np.linalg.norm(D, axis=1).max() * 1000:.1f} mm")
                print(f"  mean bias (mm)  : [{D.mean(axis=0)[0]*1000:+.1f}, "
                      f"{D.mean(axis=0)[1]*1000:+.1f}, "
                      f"{D.mean(axis=0)[2]*1000:+.1f}]")
                print(f"  std  per axis   : [{D.std(axis=0)[0]*1000:.1f}, "
                      f"{D.std(axis=0)[1]*1000:.1f}, "
                      f"{D.std(axis=0)[2]*1000:.1f}] mm")
                print()
                print("  Interpretation:")
                print("   - Large uniform bias (mean >> std) = constant offset"
                      " (translation wrong or TCP offset).")
                print("   - Large std (ratios vary) = rotation / scaling"
                      " error in T_cam_to_base.")

            # --- Candidate recalibration using the detections themselves ---
            if len(results) >= 3 and len(deltas) == len(GT_LABELS):
                print()
                print("=== Candidate recalibration from 4 detections ===")
                src_pts = []
                dst_pts = []
                used2 = set()
                pair_labels = []
                for label in GT_LABELS:
                    g = gt[label]
                    best_i = -1
                    best_d = float("inf")
                    for i, r in enumerate(results):
                        if i in used2:
                            continue
                        dist = float(np.linalg.norm(r["xyz_base"] - g))
                        if dist < best_d:
                            best_d = dist
                            best_i = i
                    if best_i < 0:
                        continue
                    used2.add(best_i)
                    r = results[best_i]
                    u, v = r["pix"]
                    Z = r["depth_mm"] / 1000.0
                    intr = cam.intrinsics
                    X = (u - intr["cx"]) * Z / intr["fx"]
                    Y = (v - intr["cy"]) * Z / intr["fy"]
                    src_pts.append([X, Y, Z])
                    dst_pts.append(g.tolist())
                    pair_labels.append(label)

                if len(src_pts) >= 3:
                    src = np.array(src_pts)
                    dst = np.array(dst_pts)
                    mu_src = src.mean(axis=0)
                    mu_dst = dst.mean(axis=0)
                    S = src - mu_src
                    Dd = dst - mu_dst
                    H = S.T @ Dd / len(src)
                    U, _, Vt = np.linalg.svd(H)
                    Wd = np.eye(3)
                    Wd[2, 2] = np.linalg.det(Vt.T @ U.T)
                    R_new = Vt.T @ Wd @ U.T
                    t_new = mu_dst - R_new @ mu_src
                    T_new = np.eye(4)
                    T_new[:3, :3] = R_new
                    T_new[:3, 3] = t_new

                    pred = (R_new @ src.T).T + t_new
                    res_new = np.linalg.norm(pred - dst, axis=1)
                    print(f"  pairs used     : {pair_labels}")
                    for lbl, r_mm in zip(pair_labels, res_new * 1000):
                        print(f"    {lbl:8s} residual = {r_mm:6.1f} mm")
                    print(f"  mean residual  : {res_new.mean() * 1000:.1f} mm")
                    print(f"  max  residual  : {res_new.max() * 1000:.1f} mm")
                    print(f"  RMS  residual  : "
                          f"{np.sqrt((res_new ** 2).mean()) * 1000:.1f} mm")
                    print()
                    print("  T_cam_to_base (candidate, from detections):")
                    for row in T_new:
                        print("    [" + ", ".join(f"{v:+.4f}"
                                                   for v in row) + "]")
                    if "--save" in sys.argv:
                        bak = CALIB_FILE.replace(
                            ".json",
                            "_bak_" + time.strftime("%Y%m%d_%H%M%S") + ".json",
                        )
                        try:
                            with open(CALIB_FILE) as f:
                                old = f.read()
                            with open(bak, "w") as f:
                                f.write(old)
                            print(f"\n  [save] old calibration backed up to "
                                  f"{bak}")
                        except Exception as ex:
                            print(f"\n  [warn] backup failed: {ex}")

                        from datetime import datetime as _dt
                        out = {
                            "timestamp": _dt.now().isoformat(timespec="seconds"),
                            "source": "analyze_static.py (fruit detections)",
                            "intrinsics": cam.intrinsics,
                            "n_points": int(len(src_pts)),
                            "labels": pair_labels,
                            "residuals_mm": [float(r * 1000) for r in res_new],
                            "mean_residual_mm": float(res_new.mean() * 1000),
                            "max_residual_mm": float(res_new.max() * 1000),
                            "rms_residual_mm":
                                float(np.sqrt((res_new ** 2).mean()) * 1000),
                            "T_cam_to_base": T_new.tolist(),
                        }
                        with open(CALIB_FILE, "w") as f:
                            json.dump(out, f, indent=2)
                        print(f"  [save] wrote candidate T to {CALIB_FILE}")
                    else:
                        print()
                        print("  NOT written to calibration.json - pass "
                              "--save to overwrite (old one backed up).")

        finally:
            cam.close()

    finally:
        try:
            q.card.close()
            q._connected = False
        except Exception:
            pass
        print("QArm released (held at last pose).")


if __name__ == "__main__":
    main()
