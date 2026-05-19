"""
Hand-eye calibration for the RealSense D415 mounted near the QArm workspace.

Workflow:
  1. You first use teach_points.py to record calibration points. Supported
     label schemes: cal_01..cal_NN (classic) or point1..pointN (short form,
     e.g. point1..point4). Each point has joints + xyz in the robot base
     frame. Place the gripper tip at a visible location on the table
     (e.g., a small marker) and save.
  2. Run this script. If both label schemes exist in teach_points.json,
     it prompts in the terminal for which group to use; if only one group
     has >= 3 points it is auto-selected. For every point in the chosen
     group:
       - Captures one RGB + depth frame from the D415.
       - Shows the RGB frame and asks you to click the same marker location.
       - Reads depth at that pixel, back-projects to camera frame.
  3. Solves the rigid transform T_cam_to_base (4x4) via the Umeyama method
     (closed-form SVD, no RANSAC needed since the points are hand-picked).
  4. Writes calibration.json with the 4x4 matrix, residual errors, and
     intrinsics used.

Notes:
  - Requires cv2 (for click capture) and the camera opens / streams via the
    Quanser SDK Python bindings.
  - At least 3 non-collinear points are required; 6+ is recommended for a
    robust fit.
  - The solver assumes rigid transform (R, t). No scale, no shear.
  - If depth at the clicked pixel is zero/invalid, the script falls back
    to the median depth in a 5x5 patch around the click; if that is still
    invalid the point is skipped.
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python (cv2) is required for the click UI.")
    sys.exit(1)

from camera import QArmCamera
from qarm_driver import QArmDriver
from qarm_kinematics import forward_kinematics, inverse_kinematics


POINTS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "teach_points.json")
)
CALIB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "calibration.json")
)

DEPTH_PATCH = 5  # pixels: half-size of the fallback depth window
MOVE_DURATION = 3.0  # seconds to interpolate between calibration points
MOVE_STEPS = 150     # interpolation steps per move
SETTLE_TIME = 1.0    # seconds to wait after arriving at a point


POINT_RE = re.compile(r"^point\d+$")
CAL_RE = re.compile(r"^cal_\d+$")

GROUPS = [
    ("cal_*", CAL_RE),
    ("point", POINT_RE),
]

MIN_CAL_POINTS = 3


def load_all_points():
    if not os.path.exists(POINTS_FILE):
        print(f"ERROR: {POINTS_FILE} not found. Run teach_points.py first.")
        sys.exit(1)
    with open(POINTS_FILE, "r") as f:
        return json.load(f)


def choose_group(all_pts):
    """Detect label groups in teach_points.json and let the user pick one.

    Returns a dict {label: point} with the chosen group's points.
    Auto-selects if only one group has enough points. Exits if none do.
    """
    available = []
    for name, pattern in GROUPS:
        pts = {k: v for k, v in all_pts.items() if pattern.match(k)}
        if len(pts) >= MIN_CAL_POINTS:
            available.append((name, pts))

    if not available:
        print(f"ERROR: no calibration group has >= {MIN_CAL_POINTS} points.")
        print("Record points labeled cal_01, cal_02, ... or point1, "
              "point2, ... with teach_points.py first.")
        sys.exit(1)

    if len(available) == 1:
        name, pts = available[0]
        print(f"Only one group available: {name} ({len(pts)} points)")
        return pts

    print("\nGroups found:")
    for i, (name, pts) in enumerate(available, start=1):
        print(f"  {i}) {name:8s} ({len(pts)} points)")
    while True:
        try:
            choice = input(f"Choose [1-{len(available)}]: ").strip()
        except EOFError:
            sys.exit(1)
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            name, pts = available[int(choice) - 1]
            print(f"Using group {name} ({len(pts)} points)")
            return pts
        print("  invalid, try again")


def colorize_depth(depth_u16):
    """Convert depth uint16 to a JET colormap for display."""
    d = depth_u16.astype(np.float32)
    if d.max() <= 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    d = np.clip(d / 1500.0, 0, 1)  # 0..1.5 m range (close workspace)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


class FrameClicker:
    def __init__(self, window_name="calibration"):
        self.window_name = window_name
        self.click_xy = None

    def _cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_xy = (x, y)

    def pick(self, cam, title_overlay):
        """Live camera view — click on the gripper tip, then Enter to confirm.

        Parameters
        ----------
        cam : QArmCamera
            Open camera (frames are read live).
        title_overlay : str
            Text shown at top of window.

        Returns
        -------
        (u, v) | "skip" | "abort"
        """
        self.click_xy = None
        self._depth = None  # depth frame captured at moment of click
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._cb)
        show_depth = False
        while True:
            # Read live frame
            try:
                color_img, depth_img = cam.read()
            except Exception:
                continue

            if show_depth:
                disp = colorize_depth(depth_img)
            else:
                disp = color_img.copy()
            cv2.putText(disp, title_overlay, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(disp, "LMB: pick   d: depth   s: skip   q: abort   ENTER: confirm",
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1)
            if self.click_xy is not None:
                u, v = self.click_xy
                cv2.circle(disp, (u, v), 6, (0, 255, 0), 2)
                # Show depth value at click from live depth
                h, w = depth_img.shape[:2]
                if 0 <= v < h and 0 <= u < w:
                    d_mm = int(depth_img[v, u])
                else:
                    d_mm = 0
                d_label = f"depth={d_mm} mm" if d_mm > 0 else "NO DEPTH"
                clr = (0, 255, 0) if d_mm > 0 else (0, 0, 255)
                cv2.putText(disp, d_label, (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
                # Store the depth frame at moment of click for later use
                self._depth = depth_img.copy()
            cv2.imshow(self.window_name, disp)
            k = cv2.waitKey(20) & 0xFF
            if k == ord("q"):
                cv2.destroyWindow(self.window_name)
                return "abort", None
            if k == ord("s"):
                cv2.destroyWindow(self.window_name)
                return "skip", None
            if k == ord("d"):
                show_depth = not show_depth
            if k in (13, 32) and self.click_xy is not None:
                # Capture final depth at the exact click position
                final_depth = self._depth if self._depth is not None else depth_img.copy()
                cv2.destroyWindow(self.window_name)
                return self.click_xy, final_depth


def depth_at(depth_img, u, v, patch=DEPTH_PATCH):
    """Return depth in metres at (u,v). Falls back to patch median if
    the exact pixel is zero."""
    h, w = depth_img.shape[:2]
    if 0 <= v < h and 0 <= u < w:
        d = int(depth_img[v, u])
        if d > 0:
            return d / 1000.0
    u0, u1 = max(0, u - patch), min(w, u + patch + 1)
    v0, v1 = max(0, v - patch), min(h, v + patch + 1)
    window = depth_img[v0:v1, u0:u1]
    valid = window[window > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0


def pixel_to_camera(u, v, depth_m, intr):
    Z = depth_m
    X = (u - intr["cx"]) * Z / intr["fx"]
    Y = (v - intr["cy"]) * Z / intr["fy"]
    return np.array([X, Y, Z], dtype=float)


def umeyama(src, dst):
    """Closed-form rigid transform R, t such that dst ≈ R @ src + t.
    Umeyama 1991 without scale. src, dst are N×3."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    S = src - mu_src
    D = dst - mu_dst
    H = S.T @ D / n
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    W = np.eye(3)
    W[2, 2] = d  # reflection-safe
    R = Vt.T @ W @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def residuals(src, dst, R, t):
    pred = (R @ src.T).T + t
    err = pred - dst
    per_pt = np.linalg.norm(err, axis=1)
    return per_pt


SAFE_Z = 0.25  # metres — altitude for safe lateral moves


def move_arm_to(qarm, target_joints, target_grip, duration=MOVE_DURATION,
                steps=MOVE_STEPS):
    """Interpolate the arm from its current position to target joints."""
    current_joints, current_grip = qarm.read_all()
    for i in range(1, steps + 1):
        alpha = i / steps
        phi = current_joints + alpha * (target_joints - current_joints)
        g = current_grip + alpha * (target_grip - current_grip)
        qarm.set_joints_and_gripper(phi, g)
        time.sleep(duration / steps)


def move_arm_safe(qarm, target_joints, target_grip):
    """Move to a calibration point via a safe waypoint above it.

    1. Compute a waypoint at the target's XY but at SAFE_Z (high).
    2. Move from current position to that waypoint (lateral move stays high).
    3. Descend from waypoint to the actual target (vertical only).
    """
    target_xyz, _ = forward_kinematics(target_joints)

    # Waypoint: same XY, safe height
    above_xyz = target_xyz.copy()
    above_xyz[2] = SAFE_Z
    try:
        above_joints = inverse_kinematics(above_xyz, gamma=0.0)
        above_joints = np.array(above_joints, dtype=float)
    except Exception:
        # IK failed for waypoint — fall back to direct move (slower)
        print("   [warn] waypoint IK failed, using slow direct move")
        move_arm_to(qarm, target_joints, target_grip,
                    duration=MOVE_DURATION * 2, steps=MOVE_STEPS * 2)
        return

    # Step 1: go to waypoint above target (safe lateral move)
    print("   -> waypoint above target...")
    move_arm_to(qarm, above_joints, target_grip,
                duration=MOVE_DURATION, steps=MOVE_STEPS)
    # Step 2: descend to actual target (short vertical drop)
    print("   -> descending to target...")
    move_arm_to(qarm, target_joints, target_grip,
                duration=2.0, steps=100)


def ascend_first(qarm):
    """Lift the arm straight up to SAFE_Z before any lateral move."""
    current_joints, _ = qarm.read_all()
    current_xyz, _ = forward_kinematics(current_joints)
    if current_xyz[2] >= SAFE_Z - 0.01:
        return  # already high enough
    above_xyz = current_xyz.copy()
    above_xyz[2] = SAFE_Z
    try:
        above_joints = inverse_kinematics(above_xyz, gamma=0.0)
        above_joints = np.array(above_joints, dtype=float)
        print("   -> ascending to safe height...")
        move_arm_to(qarm, above_joints, 0.15, duration=2.0, steps=100)
    except Exception:
        pass  # IK failed, pickhome0 move will handle it


def load_pickhome0():
    """Load pickhome0 joints from teach_points.json as safe transit position."""
    if not os.path.exists(POINTS_FILE):
        return None
    with open(POINTS_FILE, "r") as f:
        all_pts = json.load(f)
    if "pickhome0" in all_pts:
        return np.array(all_pts["pickhome0"]["joints_rad"], dtype=float)
    return None


def main():
    all_pts = load_all_points()
    cal = choose_group(all_pts)
    print(f"Loaded {len(cal)} calibration points: {sorted(cal.keys())}")
    pickhome0_joints = load_pickhome0()
    if pickhome0_joints is None:
        print("[warn] pickhome0 not found in teach_points.json — "
              "will home between points instead")

    # --- Connect QArm ---
    print("Connecting to QArm...")
    qarm = QArmDriver()
    qarm.connect()
    print("QArm connected. Homing...")
    qarm.home(duration=4.0, steps=200)
    time.sleep(0.5)

    cam = QArmCamera()
    cam.open()
    intr = cam.intrinsics
    print(f"Camera intrinsics: fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}")
    try:
        # Warmup camera
        for _ in range(10):
            try:
                color, depth = cam.read()
                if color.mean() > 5:
                    break
            except Exception:
                pass
            time.sleep(0.05)

        clicker = FrameClicker()
        src_pts = []  # camera frame (X,Y,Z) meters
        dst_pts = []  # robot base frame (x,y,z) meters
        used_labels = []

        for label in sorted(cal.keys()):
            pt = cal[label]
            world_xyz = np.array(pt["xyz_m"], dtype=float)
            target_joints = np.array(pt["joints_rad"], dtype=float)
            target_grip = float(pt.get("gripper", 0.15))

            # Move arm to this calibration point via safe waypoint
            print(f"\n>> {label}: moving arm to {world_xyz.round(3)} ...")
            move_arm_safe(qarm, target_joints, target_grip)
            # Hold position and let vibrations settle
            for _ in range(20):
                qarm.set_joints_and_gripper(target_joints, target_grip)
                time.sleep(0.05)

            # Live camera view — click on gripper tip
            overlay = f"{label}  base={world_xyz.round(3)}  — click gripper tip"
            print(f"   LIVE camera — click the gripper tip, then Enter  (d: depth)")
            result, depth_snap = clicker.pick(cam, overlay)
            if result == "abort":
                print("  user aborted"); break
            if result == "skip":
                print(f"  skipped {label}")
                if pickhome0_joints is not None:
                    ascend_first(qarm)
                    print("   returning to pickhome0...")
                    move_arm_to(qarm, pickhome0_joints, 0.15)
                continue

            u, v = result
            d_m = depth_at(depth_snap, u, v)
            if d_m is None or d_m <= 0:
                print(f"  [{label}] no valid depth at ({u},{v})  (skipped)")
                if pickhome0_joints is not None:
                    ascend_first(qarm)
                    print("   returning to pickhome0...")
                    move_arm_to(qarm, pickhome0_joints, 0.15)
                continue

            p_cam = pixel_to_camera(u, v, d_m, intr)
            print(f"  pixel=({u},{v})  depth={d_m:.3f}m  "
                  f"p_cam={p_cam.round(3)}")
            src_pts.append(p_cam)
            dst_pts.append(world_xyz)
            used_labels.append(label)

            # Return to pickhome0 before moving to next point
            if pickhome0_joints is not None:
                print("   returning to pickhome0...")
                move_arm_to(qarm, pickhome0_joints, 0.15)

        cv2.destroyAllWindows()
    finally:
        cam.close()
        print("Returning arm home...")
        try:
            qarm.home(duration=3.0)
        except Exception:
            pass
        try:
            qarm.disconnect()
        except Exception:
            pass

    if len(src_pts) < 3:
        print(f"\nERROR: need at least 3 valid pairs, got {len(src_pts)}.")
        sys.exit(1)

    src = np.array(src_pts)
    dst = np.array(dst_pts)
    R, t = umeyama(src, dst)
    res = residuals(src, dst, R, t)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    print("\n=== Calibration result ===")
    for lbl, r_mm in zip(used_labels, res * 1000):
        print(f"  {lbl:8s}  residual = {r_mm:6.1f} mm")
    print(f"  mean   residual : {res.mean() * 1000:.1f} mm")
    print(f"  max    residual : {res.max() * 1000:.1f} mm")
    print(f"  RMS    residual : "
          f"{np.sqrt((res ** 2).mean()) * 1000:.1f} mm")
    print("  T_cam_to_base (4x4):")
    for row in T:
        print("    [" + ", ".join(f"{v:+.4f}" for v in row) + "]")

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "intrinsics": intr,
        "n_points": int(len(src_pts)),
        "labels": used_labels,
        "residuals_mm": [float(r * 1000) for r in res],
        "mean_residual_mm": float(res.mean() * 1000),
        "max_residual_mm": float(res.max() * 1000),
        "rms_residual_mm": float(np.sqrt((res ** 2).mean()) * 1000),
        "T_cam_to_base": T.tolist(),
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ok] wrote {CALIB_FILE}")

    if out["max_residual_mm"] > 15:
        print("\n[warn] max residual > 15 mm — consider redoing calibration "
              "with more points or better marker visibility.")


if __name__ == "__main__":
    main()
