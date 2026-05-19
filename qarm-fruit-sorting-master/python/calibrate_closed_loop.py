"""Closed-loop UGreen-driven hand-eye calibration.

Workflow (hardware session):
  1. Load teach points and UGreen intrinsics.
  2. Capture baseline frame (arm out of UGreen view — user moves arm to
     homeplace0 / pickhome0 first; baseline is whatever the UGreen sees).
  3. For each of N calibration labels (cal_*):
       - Move arm to that teach point.
       - Capture UGreen frame.
       - tcp_from_diff -> pixel (u, v).
       - Pair with the teach point xyz (base-frame ground truth).
  4. Solve PnP (cv2.solvePnP) for T_base_to_cam, then invert for
     T_cam_to_base. Write calibration_ugreen.json with intrinsics +
     transform + residuals.

The resulting T_cam_to_base lets us convert any UGreen pixel that lies
on the known table plane (z ~= 0.13 m) to a base-frame XY — independent
cross-check of the D415 calibration.
"""
import datetime
import json
import os
import sys
import time
import numpy as np
import cv2

from qarm_driver import QArmDriver
from ugreen_tracker import capture, tcp_from_diff, save_baseline, load_baseline
from ugreen_intrinsics import load_intrinsics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
POINTS_FILE = os.path.join(REPO, "teach_points.json")
CALIB_OUT = os.path.join(REPO, "calibration_ugreen.json")
CALIB_REJECTED = os.path.join(REPO, "calibration_ugreen.rejected.json")
BASELINE_PATH = os.path.join(REPO, "logs", "ugreen_baseline.png")
CAPTURE_DIR = os.path.join(REPO, "logs", "closed_loop")

# Sanity gates on the base-plane residual (metres). Beyond MAX we refuse
# to overwrite the real calibration file; between WARN and MAX we still
# write it but flag the result so downstream scripts can notice.
CALIB_MAX_RMS_M = 0.050      # > 50 mm => write rejected file, not the real one
CALIB_WARN_RMS_M = 0.020     # > 20 mm => print a warning but still write


def solve_extrinsics(base_pts, pixels, K, dist, z_table=0.13):
    """Solve cv2.solvePnP for T_cam_to_base given N correspondences.

    Parameters
    ----------
    base_pts : (N, 3) float — object points in the robot base frame.
    pixels   : (N, 2) float — image points in the UGreen pixel frame.
    K        : (3, 3) float — camera intrinsic matrix (pixels).
    dist     : (5,)   float — distortion coefficients.
    z_table  : float         — table-plane height in base frame (m).

    Returns
    -------
    T_cam_to_base : (4, 4) float
    residuals     : dict with keys
        'rms_px'             raw reprojection pixel RMS
        'rms_base_m'         unprojection-to-z_table plane residual RMS (m),
                             or None if fewer than 2 valid intersections
        'max_base_m'         worst-case base-plane residual (m), or None
        'per_label_base_m'   list of per-point base-plane residuals (m);
                             np.nan where the ray is nearly parallel to the
                             plane.
    """
    if len(base_pts) < 4:
        raise ValueError("need >= 4 correspondences")

    base_pts = np.asarray(base_pts, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    # Primary: SQPnP (globally optimal, robust to near-coplanar geometry
    # which is exactly our case — cal_01..cal_08 all sit on the table).
    ok, rvec, tvec = cv2.solvePnP(
        base_pts.reshape(-1, 1, 3),
        pixels.reshape(-1, 1, 2),
        K, dist,
        flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("cv2.solvePnP (SQPNP) returned False")

    # Refine: ITERATIVE with the SQPnP estimate as extrinsic guess.
    # The DLT init that ITERATIVE uses on its own is weak for coplanar
    # inputs; seeded with a good guess it just polishes.
    ok2, rvec, tvec = cv2.solvePnP(
        base_pts.reshape(-1, 1, 3),
        pixels.reshape(-1, 1, 2),
        K, dist,
        rvec=rvec, tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok2:
        raise RuntimeError("cv2.solvePnP (ITERATIVE refine) returned False")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    # R,t give base -> cam. Invert for cam -> base.
    T_b2c = np.eye(4); T_b2c[:3, :3] = R; T_b2c[:3, 3] = t
    T_c2b = np.linalg.inv(T_b2c)

    # ---- Pixel residuals ------------------------------------------------
    reproj_pix, _ = cv2.projectPoints(
        base_pts.reshape(-1, 1, 3), rvec, tvec, K, dist)
    err_px = np.linalg.norm(pixels - reproj_pix.reshape(-1, 2), axis=1)
    rms_px = float(np.sqrt((err_px ** 2).mean()))

    # ---- Base-plane residuals ------------------------------------------
    # For each pixel: undistort to normalised camera coords, build the
    # viewing ray, transform to base frame using T_cam_to_base, intersect
    # with z = z_table, compare XY to the teach-point XY.
    pix_nd = pixels.reshape(-1, 1, 2).astype(np.float64)
    norm_pts = cv2.undistortPoints(pix_nd, K, dist).reshape(-1, 2)
    R_c2b = T_c2b[:3, :3]
    origin_base = T_c2b[:3, 3]

    per_label_base = []
    for i, (xn, yn) in enumerate(norm_pts):
        ray_cam = np.array([xn, yn, 1.0])
        ray_base = R_c2b @ ray_cam
        if abs(ray_base[2]) < 1e-6:
            # Ray nearly parallel to the table plane — degenerate.
            per_label_base.append(float('nan'))
            continue
        s = (z_table - origin_base[2]) / ray_base[2]
        hit = origin_base + s * ray_base
        err_xy = np.linalg.norm(hit[:2] - base_pts[i, :2])
        per_label_base.append(float(err_xy))

    per_label_arr = np.array(per_label_base, dtype=np.float64)
    valid = np.isfinite(per_label_arr)
    if valid.sum() < 2:
        rms_base_m = None
        max_base_m = None
    else:
        rms_base_m = float(np.sqrt((per_label_arr[valid] ** 2).mean()))
        max_base_m = float(per_label_arr[valid].max())

    residuals = {
        'rms_px': rms_px,
        'rms_base_m': rms_base_m,
        'max_base_m': max_base_m,
        'per_label_base_m': per_label_base,
    }
    return T_c2b, residuals


def slow_move_to_joints(q, target, grip, seconds=3.0, steps=200):
    cj, cg = q.read_all()
    cj = np.array(cj, dtype=float); cg = float(cg)
    for i in range(1, steps + 1):
        a = i / steps
        s = 3 * a * a - 2 * a * a * a
        q.set_joints_and_gripper(
            cj + s * (target - cj),
            cg + s * (grip - cg))
        time.sleep(seconds / steps)
    for _ in range(20):
        q.set_joints_and_gripper(target, grip)
        time.sleep(0.05)


def run_calibration(labels, baseline_path=BASELINE_PATH,
                     capture_dir=CAPTURE_DIR):
    """Execute the closed-loop calibration against the labels in order.
    Returns the result dict and writes calibration_ugreen.json (or
    calibration_ugreen.rejected.json if the sanity gate fires)."""
    os.makedirs(capture_dir, exist_ok=True)

    with open(POINTS_FILE) as f:
        pts = json.load(f)
    intr, dist = load_intrinsics()
    K = np.array([[intr['fx'], 0, intr['cx']],
                  [0, intr['fy'], intr['cy']],
                  [0, 0, 1]], dtype=np.float64)

    baseline = load_baseline(baseline_path)

    q = QArmDriver(); q.connect(); time.sleep(0.3)
    for _ in range(5):
        try:
            q.read_all(); break
        except Exception:
            time.sleep(0.3)

    base_pts, pixels, used_labels = [], [], []
    try:
        for lbl in labels:
            if lbl not in pts:
                print(f"  [skip] {lbl} missing from teach_points"); continue
            target = np.array(pts[lbl]['joints_rad'], dtype=float)
            grip = float(pts[lbl].get('gripper', 0.15))
            print(f"\n-> {lbl}")
            slow_move_to_joints(q, target, grip)
            frame = capture()
            out = os.path.join(capture_dir, f"{lbl}.png")
            cv2.imwrite(out, frame)
            tcp = tcp_from_diff(frame, baseline)
            if tcp is None:
                print(f"    no TCP detected; skipping")
                continue
            base_pts.append(pts[lbl]['xyz_m'])
            pixels.append(list(tcp))
            used_labels.append(lbl)
            print(f"    pixel={tcp}  base={np.round(pts[lbl]['xyz_m'],3)}")
    finally:
        try: q.card.close()
        except Exception: pass
        q._connected = False

    if len(used_labels) < 4:
        raise RuntimeError(f"only {len(used_labels)} correspondences, need >= 4")

    base_pts = np.array(base_pts, dtype=np.float64)
    pixels = np.array(pixels, dtype=np.float64)
    T_c2b, residuals = solve_extrinsics(base_pts, pixels, K, dist)

    rms_px = residuals['rms_px']
    rms_base_m = residuals['rms_base_m']
    max_base_m = residuals['max_base_m']

    # ---- Sanity gate ----------------------------------------------------
    # ANSI colour codes: keep it simple, operators see raw terminal.
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    reason = None
    if rms_base_m is None:
        status = 'rejected'
        reason = 'fewer than 2 valid base-plane intersections'
    elif rms_base_m > CALIB_MAX_RMS_M:
        status = 'rejected'
        reason = (f'rms_base_m {rms_base_m * 1000:.1f} mm > '
                  f'{CALIB_MAX_RMS_M * 1000:.0f} mm hard limit')
    elif rms_base_m > CALIB_WARN_RMS_M:
        status = 'warning'
        reason = (f'rms_base_m {rms_base_m * 1000:.1f} mm > '
                  f'{CALIB_WARN_RMS_M * 1000:.0f} mm warn threshold')
    else:
        status = 'ok'

    out = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'source': 'calibrate_closed_loop.py',
        'camera': 'UGreen',
        'intrinsics': intr,
        'dist': list(map(float, dist.flatten())),
        'n_points': int(len(used_labels)),
        'labels': used_labels,
        'rms_px': float(rms_px),
        'rms_base_m': None if rms_base_m is None else float(rms_base_m),
        'max_base_m': None if max_base_m is None else float(max_base_m),
        'per_label_base_m': residuals['per_label_base_m'],
        'status': status,
        'T_cam_to_base': T_c2b.tolist(),
    }
    if reason is not None:
        out['reason'] = reason

    # ---- Printed summary -----------------------------------------------
    print()
    print(f"RMS pixel : {rms_px:.2f} px")
    if rms_base_m is None:
        print("RMS base  : n/a (insufficient valid intersections)")
        print("Max base  : n/a")
    else:
        print(f"RMS base  : {rms_base_m * 1000:.1f} mm  (target < 15 mm)")
        print(f"Max base  : {max_base_m * 1000:.1f} mm")
    print(f"status    : {status}")

    if status == 'rejected':
        with open(CALIB_REJECTED, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"{RED}"
              f"###############################################################\n"
              f"# CALIBRATION REJECTED — {reason}\n"
              f"# NOT writing {CALIB_OUT}\n"
              f"# Wrote diagnostic dump: {CALIB_REJECTED}\n"
              f"###############################################################"
              f"{RESET}")
        return out

    with open(CALIB_OUT, 'w') as f:
        json.dump(out, f, indent=2)

    if status == 'warning':
        print(f"{YELLOW}"
              f"WARNING: {reason}. Calibration written but verify before use."
              f"{RESET}")
    else:
        print(f"{GREEN}wrote {CALIB_OUT}{RESET}")
    return out


if __name__ == "__main__":
    default_labels = [f"cal_{i:02d}" for i in range(1, 9)]
    labels = sys.argv[1:] if len(sys.argv) > 1 else default_labels
    run_calibration(labels)
