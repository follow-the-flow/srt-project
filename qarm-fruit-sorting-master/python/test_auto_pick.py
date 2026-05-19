"""Autonomous detect-and-pick test: D415 vision + staged approach + release.

Flow:
  1. Move arm to pickhome1
  2. D415 warmup -> single frame -> detect strawberries
  3. For each valid detection (filtered to workspace):
     a. UGreen "before" snapshot
     b. Staged approach (rotate+advance -> descend) to detected XYZ
     c. Close gripper (ramped)
     d. Staged retreat (ascend -> lateral to pickhome)
     e. UGreen "lifted" snapshot
     f. Go to 'placeberries' teach point (drop zone)
     g. Open gripper (release)
     h. Return to pickhome1
  4. Final UGreen snapshot + summary

Uses existing calibration.json (D415 T_cam_to_base) and teach_points.json.

Flags:
  --max N     only process first N detections (default: all)
  --dry-run   plan + print, do not move the arm
  --label L   pick only red blobs of type L (strawberry or tomato)
"""
import json
import os
import sys
import time
import numpy as np
import cv2

from qarm_driver import QArmDriver
from qarm_kinematics import forward_kinematics, inverse_kinematics, JOINT_LIMITS
from camera import QArmCamera
from fruit_detector import detect_fruits, draw_detections, detection_depth_mm
from ugreen_tracker import capture

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS_FILE = os.path.join(REPO, "teach_points.json")
CALIB_FILE = os.path.join(REPO, "calibration.json")
OUT_DIR = os.path.join(REPO, "logs", "auto_pick_test")

PICKHOME = "pickhome1"
RELEASE_LABEL = "placeberries"

GRIP_OPEN = 0.2
GRIP_CLOSE = 0.9

# Workspace filter — reject detections outside this 3D box
WS_X = (0.15, 0.65)
WS_Y = (-0.35, 0.35)
WS_Z = (-0.05, 0.30)

TRANSIT_DZ = 0.15


def get_flag(name):
    return name in sys.argv


def get_arg(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            try:
                return cast(sys.argv[i + 1])
            except ValueError:
                pass
    return default


def slow_move(q, target_joints, target_grip, seconds=3.0, steps=200):
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
    for _ in range(15):
        q.set_joints_and_gripper(target_joints, target_grip)
        time.sleep(0.05)


def ramp_gripper(q, joints, start, end, steps=25, dt=0.04):
    for a in np.linspace(start, end, steps):
        q.set_joints_and_gripper(joints, float(a))
        time.sleep(dt)
    for _ in range(10):
        q.set_joints_and_gripper(joints, end)
        time.sleep(0.05)


def ik_safe(xyz, gamma=0.0):
    try:
        phi = inverse_kinematics(xyz, gamma=gamma)
    except Exception as ex:
        return None, f"IK error: {ex}"
    phi = np.array(phi, dtype=float)
    for i in range(4):
        lo, hi = JOINT_LIMITS[i]
        if phi[i] < lo or phi[i] > hi:
            return None, (f"joint {i} out of limits "
                          f"({np.degrees(phi[i]):+.1f} deg)")
    return phi, None


def approach_staged_ik(q, target_xyz, grip, seconds=2.5):
    """Rotate+advance at transit-Z, then descend vertically. Full IK
    (no pre-taught joints needed). Returns True on success."""
    cj, _ = q.read_all()
    cj = np.array(cj, dtype=float)
    cur_xyz, _ = forward_kinematics(cj)
    cur_xyz = np.array(cur_xyz, dtype=float)

    transit_z = max(cur_xyz[2], target_xyz[2] + TRANSIT_DZ)

    wp_xyz = np.array([target_xyz[0], target_xyz[1], transit_z])
    phi_wp, err = ik_safe(wp_xyz)
    if err:
        print(f"    [FAIL] waypoint IK: {err}")
        return False

    phi_target, err = ik_safe(target_xyz)
    if err:
        print(f"    [FAIL] target IK: {err}")
        return False

    print(f"    stage 1: rotate+advance to xyz={wp_xyz.round(3)}")
    slow_move(q, phi_wp, grip, seconds=seconds)
    print(f"    stage 2: descend to xyz={target_xyz.round(3)}")
    slow_move(q, phi_target, grip, seconds=seconds * 0.8)
    return True


def retreat_staged_ik(q, from_xyz, ph_joints, grip, seconds=2.5):
    """Ascend straight up from current xyz, then lateral to pickhome."""
    transit_xyz = np.array([from_xyz[0], from_xyz[1],
                             from_xyz[2] + TRANSIT_DZ])
    phi_up, err = ik_safe(transit_xyz)
    if err:
        print(f"    [warn] retreat IK fail: {err}, direct to pickhome")
        slow_move(q, ph_joints, grip, seconds=seconds * 1.5)
        return
    print(f"    stage 1: ascend to xyz={transit_xyz.round(3)}")
    slow_move(q, phi_up, grip, seconds=seconds * 0.8)
    print(f"    stage 2: lateral to pickhome1")
    slow_move(q, ph_joints, grip, seconds=seconds)


def snap(tag):
    frame = capture()
    path = os.path.join(OUT_DIR, f"{tag}.png")
    cv2.imwrite(path, frame)
    return path


def detect_red(cam, T_cam_to_base, filter_type=None):
    """Grab one stable frame, detect red fruits, project to base frame,
    filter to workspace. Returns (color, depth, list[dict])."""
    for _ in range(5):
        try: cam.read()
        except Exception: pass
        time.sleep(0.05)
    color, depth = cam.read()
    dets = detect_fruits(color, depth)
    out = []
    for d in dets:
        if d.fruit_type not in ("strawberry", "tomato"):
            continue
        if filter_type and d.fruit_type != filter_type:
            continue
        dmm = detection_depth_mm(d, depth)
        if dmm is None:
            continue
        row, col = d.centroid
        xyz = cam.pixel_to_world(row, col, dmm, T_cam_to_base)
        if not (WS_X[0] <= xyz[0] <= WS_X[1]):
            print(f"  [filter] {d.fruit_type} at {np.round(xyz,3)} out of X range")
            continue
        if not (WS_Y[0] <= xyz[1] <= WS_Y[1]):
            print(f"  [filter] {d.fruit_type} at {np.round(xyz,3)} out of Y range")
            continue
        if not (WS_Z[0] <= xyz[2] <= WS_Z[1]):
            print(f"  [filter] {d.fruit_type} at {np.round(xyz,3)} out of Z range")
            continue
        out.append({
            "type": d.fruit_type,
            "pix": (int(col), int(row)),
            "depth_mm": dmm,
            "xyz": np.array(xyz, dtype=float),
            "area": int(d.area),
        })
    # Sort by Y descending (reach near-side first) - arbitrary but stable
    out.sort(key=lambda h: -h["xyz"][1])
    return color, depth, dets, out


def main():
    max_fruits = get_arg("--max", None, int)
    dry_run = get_flag("--dry-run")
    filter_type = get_arg("--label", None)

    with open(POINTS_FILE) as f:
        pts = json.load(f)
    with open(CALIB_FILE) as f:
        cal = json.load(f)
    T = np.array(cal["T_cam_to_base"], dtype=float)
    rms = cal.get("rms_residual_mm", "?")
    print(f"Calibration: RMS={rms} mm  ({cal.get('n_points', '?')} points)")

    for key in (PICKHOME, RELEASE_LABEL):
        if key not in pts:
            print(f"ERROR: teach point '{key}' missing in {POINTS_FILE}")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    ph_joints = np.array(pts[PICKHOME]["joints_rad"], dtype=float)
    rel_joints = np.array(pts[RELEASE_LABEL]["joints_rad"], dtype=float)
    rel_grip = float(pts[RELEASE_LABEL].get("gripper", GRIP_OPEN))

    q = QArmDriver()
    q.connect()
    time.sleep(0.5)
    for _ in range(5):
        try: q.read_all(); break
        except Exception: time.sleep(0.3)

    cam = QArmCamera()
    cam.open()

    picks = []
    fails = []

    try:
        print(f"\nMoving to {PICKHOME}...")
        slow_move(q, ph_joints, GRIP_OPEN, seconds=3.0)

        # warmup
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                c, d = cam.read()
                if c.mean() > 5 and (d > 0).mean() > 0.10:
                    break
            except Exception: pass
            time.sleep(0.1)

        print("Detecting red fruits...")
        color, depth, all_dets, targets = detect_red(cam, T, filter_type)
        scene_path = os.path.join(OUT_DIR, "00_scene_detections.png")
        cv2.imwrite(scene_path, draw_detections(color, all_dets))
        print(f"  scene -> {scene_path}")
        print(f"  total detections: {len(all_dets)}  "
              f"valid in-workspace: {len(targets)}")

        if not targets:
            print("No targets. Exiting.")
            return

        if max_fruits is not None:
            targets = targets[:max_fruits]
            print(f"  limited to first {len(targets)} by --max")

        snap("00_initial")

        for i, t in enumerate(targets, 1):
            xyz = t["xyz"]
            print(f"\n--- {i}/{len(targets)}: {t['type']} at xyz={xyz.round(3)} "
                  f"pix={t['pix']} depth={t['depth_mm']:.0f}mm area={t['area']} ---")

            if dry_run:
                print("  [dry-run] skip motion")
                continue

            snap(f"{i:02d}_{t['type']}_01_before")

            print(f"  [1] approach {t['type']} (staged)...")
            ok = approach_staged_ik(q, xyz, GRIP_OPEN)
            if not ok:
                print(f"  [skip] approach failed")
                fails.append({"idx": i, "xyz": xyz.tolist(), "reason": "approach IK"})
                continue

            print(f"  [2] close gripper...")
            _, phi_now = q.read_all(), None  # noop to silence linter
            current_joints, _ = q.read_all()
            current_joints = np.array(current_joints, dtype=float)
            ramp_gripper(q, current_joints, GRIP_OPEN, GRIP_CLOSE)
            time.sleep(0.3)
            _, gr = q.read_all()
            print(f"      gripper readback: {gr:.3f}  "
                  f"({'holding?' if gr < 0.95 else 'empty'})")

            print(f"  [3] retreat to pickhome1...")
            retreat_staged_ik(q, xyz, ph_joints, GRIP_CLOSE)
            time.sleep(0.5)
            snap(f"{i:02d}_{t['type']}_02_lifted")

            print(f"  [4] transit to {RELEASE_LABEL}...")
            slow_move(q, rel_joints, GRIP_CLOSE, seconds=3.0)
            time.sleep(0.3)
            snap(f"{i:02d}_{t['type']}_03_atrelease")

            print(f"  [5] open gripper (release)...")
            ramp_gripper(q, rel_joints, GRIP_CLOSE, GRIP_OPEN)
            time.sleep(0.3)

            print(f"  [6] return to pickhome1...")
            slow_move(q, ph_joints, GRIP_OPEN, seconds=3.0)
            snap(f"{i:02d}_{t['type']}_04_back")

            picks.append({
                "idx": i, "type": t["type"], "xyz": xyz.tolist(),
                "gr_close": float(gr),
            })

        snap("99_final")

    finally:
        try: cam.close()
        except Exception: pass
        try:
            q.card.close(); q._connected = False
        except Exception: pass
        print("\nQArm released (held at last pose).")

    # Report
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for p in picks:
        print(f"  [{p['idx']}] {p['type']:10s} xyz={np.round(p['xyz'],3)}  "
              f"gripper={p['gr_close']:.3f}")
    for f in fails:
        print(f"  [{f['idx']}] FAIL  xyz={np.round(f['xyz'],3)}  "
              f"reason={f['reason']}")
    print(f"\nUGreen frames in: {OUT_DIR}")


if __name__ == "__main__":
    main()
