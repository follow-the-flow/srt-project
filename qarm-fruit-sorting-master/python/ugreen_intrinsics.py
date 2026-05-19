"""UGreen intrinsic-camera calibration via OpenCV chessboard.

Usage (hardware session):
    C:/Python313/python.exe python/ugreen_intrinsics.py collect
        — captures 20 frames as the user shows the printed chessboard
          to the UGreen in different orientations.
    C:/Python313/python.exe python/ugreen_intrinsics.py solve
        — runs cv2.calibrateCamera and writes ugreen_intrinsics.json.

Offline, callers use load_intrinsics(path) to get the camera matrix.
"""
import json
import os
import sys
import time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_JSON = os.path.join(REPO, "ugreen_intrinsics.json")
CAPTURE_DIR = os.path.join(REPO, "logs", "ugreen_chessboards")

INNER_CORNERS = (7, 5)        # must match scripts/print_chessboard.py
SQUARE_MM = 30.0


def save_intrinsics(path, intrinsics, dist):
    payload = {
        'intrinsics': {k: float(v) for k, v in intrinsics.items()},
        'dist': [float(x) for x in np.asarray(dist).flatten()],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_intrinsics(path=DEFAULT_JSON):
    with open(path) as f:
        p = json.load(f)
    return (dict(p['intrinsics']),
            np.asarray(p['dist'], dtype=np.float64))


def collect_frames(n=20, out_dir=CAPTURE_DIR):
    """Live-preview chessboard capture. The UGreen feed is shown in an
    OpenCV window with detected corners overlaid in real time. Press
    SPACE to save the current pose (only counts if corners are detected);
    press 'q' or ESC to quit early. Requires hardware."""
    from ugreen_tracker import UGREEN_IDX
    os.makedirs(out_dir, exist_ok=True)
    detect_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                    cv2.CALIB_CB_NORMALIZE_IMAGE)
    WINDOW = "UGreen — chessboard collect  (SPACE=save, Q=quit)"

    print(f"Need {n} detected poses. Live preview in a new window.")
    print("  GREEN overlay + corners = detected; press SPACE to save.")
    print("  RED message = no pattern found; reposition and try again.")
    print("  Q or ESC = quit early.")
    print(f"[debug] opening UGreen idx={UGREEN_IDX} MSMF ...", flush=True)

    cap = cv2.VideoCapture(UGREEN_IDX, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"UGreen did not open (idx={UGREEN_IDX} MSMF)")
    # MSMF on Windows needs explicit media-type negotiation; without it
    # grabFrame returns MF_E_INVALIDMEDIATYPE (-1072875772). Force MJPG
    # FOURCC (more reliable than default YUYV at 720p), then set the
    # resolution with small sleeps between so MSMF has time to commit
    # each reconfigure.
    print("[debug] negotiating MJPG 1280x720 ...", flush=True)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        time.sleep(0.3)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        time.sleep(0.3)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(0.3)
    except Exception as ex:
        print(f"[debug] set() failed: {ex} — trying default res", flush=True)
    print("[debug] camera negotiated, grabbing warmup frame ...", flush=True)
    for _ in range(5):
        ok, test = cap.read()
        if ok and test is not None and test.size > 0:
            print(f"[debug] warmup frame shape={test.shape} "
                  f"mean={test.mean():.0f}", flush=True)
            break
    else:
        print("[debug] WARNING: no valid warmup frame", flush=True)
    print("[debug] creating preview window ...", flush=True)
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
    except Exception as ex:
        print(f"[debug] namedWindow failed: {ex}", flush=True)
        raise
    # Best-effort topmost — some OpenCV builds don't support this on Win.
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception as ex:
        print(f"[debug] topmost hint failed (non-fatal): {ex}", flush=True)
    print("[debug] entering loop — if you don't see the window, "
          "check behind the terminal (alt+tab on Windows).", flush=True)

    i = 0
    try:
        while i < n:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, INNER_CORNERS, detect_flags)

            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, INNER_CORNERS, corners, found)
                msg = f"[{i+1}/{n}] DETECTED — press SPACE to save"
                color = (0, 255, 0)
            else:
                msg = f"[{i+1}/{n}] no chessboard — reposition"
                color = (0, 0, 255)

            cv2.putText(vis, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                print(f"aborted at {i}/{n}")
                break
            if key == 32:  # SPACE
                if not found:
                    print("  [skip] no chessboard in current frame")
                    continue
                path = os.path.join(out_dir, f"chess_{i:02d}.png")
                cv2.imwrite(path, frame)
                print(f"  [{i+1}/{n}] saved {path}")
                i += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if i < n:
        print(f"collected {i}/{n} poses — re-run 'collect' to add more "
              f"or run 'solve' if satisfied.")
    else:
        print(f"done — {n}/{n} poses captured. Run 'solve' next.")


def solve(in_dir=CAPTURE_DIR, out_json=DEFAULT_JSON):
    """Run cv2.calibrateCamera on the collected chessboard images.
    Writes intrinsics + distortion to out_json."""
    files = sorted(
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir) if f.startswith("chess_"))
    if len(files) < 10:
        raise RuntimeError(
            f"need >= 10 chessboard views, got {len(files)}")

    objp = np.zeros((INNER_CORNERS[0] * INNER_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:INNER_CORNERS[0], 0:INNER_CORNERS[1]].T.reshape(-1, 2)
    objp *= SQUARE_MM / 1000.0  # convert to metres

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_list = []
    img_list = []
    gray_shape = None
    used = 0
    for f in files:
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, INNER_CORNERS,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_list.append(objp)
        img_list.append(corners)
        used += 1
    if used < 10:
        raise RuntimeError(f"only {used} usable views; re-collect")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_list, img_list, gray_shape, None, None)
    intrinsics = {
        'fx': float(K[0, 0]),
        'fy': float(K[1, 1]),
        'cx': float(K[0, 2]),
        'cy': float(K[1, 2]),
    }
    save_intrinsics(out_json, intrinsics, dist)
    print(f"RMS reprojection error: {rms:.3f} px  (target < 1.0)")
    print(f"K = {K}")
    print(f"dist = {dist.flatten()}")
    print(f"wrote {out_json}  (used {used} views)")
    return rms, intrinsics, dist


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "solve"
    if mode == "collect":
        collect_frames()
    elif mode == "solve":
        solve()
    else:
        print(__doc__); sys.exit(1)
