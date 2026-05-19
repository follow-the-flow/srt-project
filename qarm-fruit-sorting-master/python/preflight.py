"""Pre-lab-session sanity script.

Runs a handful of checks in order. Each check returns (ok, message). The
script prints a green 'PREFLIGHT OK' or red 'PREFLIGHT FAIL: <reason>'
and exits with the number of failing checks.

Checks:
  1. QArm connect + read_all + one set_joints round trip (-843 early).
  2. D415 warmup (color mean > 5, depth valid > 10%).
  3. session_cal.json loadable; RMS < 2.0 px; age < 12 h; survey pose
     present.
  4. Chessboard still visible from survey1 (homography residual small).
  5. HSV detector runs on one D415 frame; per-blob circ/sat printed.
  6. Offline tests — zero hardware dependency.

Usage:
  py -3.13 python/preflight.py            # full
  py -3.13 python/preflight.py --offline  # hardware checks skipped
"""
import os
import sys
import time
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SESSION_CAL = os.path.join(REPO, "session_cal.json")

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def _mark(ok):
    return (ANSI_GREEN + "OK  " + ANSI_RESET) if ok else (ANSI_RED + "FAIL" + ANSI_RESET)


def check_qarm():
    try:
        from qarm_driver import QArmDriver
        import numpy as np
    except Exception as ex:
        return False, f"import: {ex}"
    q = QArmDriver()
    try:
        q.connect()
        time.sleep(0.4)
        for _ in range(5):
            try:
                j, g = q.read_all(); break
            except Exception:
                time.sleep(0.3)
        else:
            return False, "read_all timeout -> power cycle QArm"
        q.set_joints_and_gripper(j, g)
    except Exception as ex:
        return False, f"driver: {ex}"
    finally:
        # Skip q.disconnect() — it homes the arm (moves to joints=0)
        # which we do NOT want during preflight. Release the HAL card
        # directly and flip the private flag so the driver cleans up.
        try: q.card.close()
        except Exception: pass
        q._connected = False
    return True, f"joints OK, gripper={g:.2f}"


def check_d415():
    try:
        from camera import QArmCamera
    except Exception as ex:
        return False, f"import: {ex}"
    cam = QArmCamera()
    try:
        cam.open()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                c, d = cam.read()
            except Exception:
                continue
            if c.mean() > 5 and (d > 0).mean() > 0.10:
                return True, (f"color mean={c.mean():.0f} "
                              f"depth={(d>0).mean()*100:.0f}% valid")
            time.sleep(0.1)
        return False, "no valid frame after 10 s"
    except Exception as ex:
        return False, f"{ex}"
    finally:
        try: cam.close()
        except Exception: pass


def check_session_cal():
    if not os.path.exists(SESSION_CAL):
        return False, (f"{os.path.basename(SESSION_CAL)} missing — "
                        f"run calibrate_chessboard.py")
    try:
        from session_cal import SessionCal
        cal = SessionCal.load(SESSION_CAL)
    except Exception as ex:
        return False, f"load: {ex}"
    rms = float(cal.homography_reproj_rms_px)
    try:
        age_h = (datetime.now()
                 - datetime.fromisoformat(cal.timestamp)
                 ).total_seconds() / 3600.0
    except Exception:
        age_h = None
    warn = []
    ok = True
    if rms >= 2.0:
        warn.append(f"RMS={rms:.2f}px >= 2.0"); ok = False
    if age_h is not None and age_h > 12:
        warn.append(f"age={age_h:.1f}h > 12"); ok = False
    if cal.survey_pose_joints_rad is None or len(cal.survey_pose_joints_rad) != 4:
        warn.append("survey_pose missing"); ok = False
    if cal.cam_extrinsics_survey1 is None:
        warn.append("cam_extrinsics_survey1 missing")
        ok = False
    else:
        pnp_rms = cal.cam_extrinsics_survey1.get("reproj_rms_px", float("inf"))
        if pnp_rms > 5.0:
            warn.append(f"solvePnP RMS={pnp_rms:.2f}px > 5.0")
            ok = False
        cam_z = cal.cam_extrinsics_survey1.get("C_cam_in_base_m", [0, 0, 0])[2]
        chess_z = float(cal.chess_origin_in_base_m[2])
        if cam_z <= chess_z:
            warn.append(
                f"camera Z {cam_z:.3f} <= chess origin Z {chess_z:.3f} "
                f"(mirror-flipped solvePnP?)")
            ok = False
    detail = f"RMS={rms:.2f}px age={age_h:.1f}h" if age_h is not None \
        else f"RMS={rms:.2f}px age=?"
    if warn:
        detail += f"  warnings={warn}"
    return ok, detail


def check_chessboard_still_visible():
    """Move arm to survey1, capture one frame, check chessboard residual."""
    try:
        from session_cal import SessionCal
        from camera import QArmCamera
        from qarm_driver import QArmDriver
        from survey_capture import _warmup_and_capture, _chessboard_residual
        from calibrate_closed_loop import slow_move_to_joints
    except Exception as ex:
        return False, f"import: {ex}"
    if not os.path.exists(SESSION_CAL):
        return False, "session_cal missing"
    try:
        cal = SessionCal.load(SESSION_CAL)
    except Exception as ex:
        return False, f"load: {ex}"
    q = QArmDriver()
    try:
        q.connect(); time.sleep(0.3)
        slow_move_to_joints(q, cal.survey_pose_joints_rad, 0.10)
    except Exception as ex:
        try: q.card.close()
        except Exception: pass
        return False, f"arm move: {ex}"
    cam = QArmCamera()
    try:
        cam.open()
        color, _, _ = _warmup_and_capture(cam)
    except Exception as ex:
        try: cam.close()
        except Exception: pass
        try: q.card.close()
        except Exception: pass
        return False, f"capture: {ex}"
    finally:
        try: cam.close()
        except Exception: pass
        try: q.card.close()
        except Exception: pass
        q._connected = False
    residual, n = _chessboard_residual(color, cal)
    if residual is None:
        return False, f"chessboard not found ({n} corners)"
    if residual > 10.0:
        return False, f"residual={residual:.1f}mm > 10 — recalibrate"
    warn = " (drift warn)" if residual > 3.0 else ""
    return True, f"residual={residual:.1f}mm corners={n}{warn}"


def check_hsv():
    try:
        from camera import QArmCamera
        from fruit_detector import detect_fruits
    except Exception as ex:
        return False, f"import: {ex}"
    cam = QArmCamera()
    try:
        cam.open()
        for _ in range(30):
            try: c, d = cam.read()
            except Exception: pass
            if c.mean() > 5 and (d > 0).mean() > 0.10: break
            time.sleep(0.1)
        dets = detect_fruits(c, d)
        lines = [f"{len(dets)} blobs"]
        for i, x in enumerate(dets):
            lines.append(
                f"  [{i}] {x.fruit_type} area={x.area:.0f} "
                f"conf={x.confidence:.2f}")
        return True, "; ".join(lines)
    except Exception as ex:
        return False, f"{ex}"
    finally:
        try: cam.close()
        except Exception: pass


def check_offline_tests():
    """Run the offline test scripts in-process and sum their exit codes."""
    test_files = [
        os.path.join(HERE, "test_integration.py"),
        os.path.join(HERE, "test_fruit_detector.py"),
        os.path.join(HERE, "test_calibrate_chessboard.py"),
        os.path.join(HERE, "test_pick_single.py"),
        os.path.join(HERE, "test_survey_capture.py"),
        os.path.join(HERE, "test_picker_viewer.py"),
    ]
    fails = 0
    names = []
    for tf in test_files:
        r = subprocess.run([sys.executable, tf],
                             capture_output=True, text=True)
        if r.returncode != 0:
            fails += 1
            names.append(os.path.basename(tf))
    if fails:
        return False, f"{fails} test file(s) failed: {names}"
    return True, "all offline tests passed"


CHECKS = [
    ("1 QArm       ", check_qarm,                      False),
    ("2 D415       ", check_d415,                      False),
    ("3 SessionCal ", check_session_cal,               True),
    ("4 Chessboard ", check_chessboard_still_visible,  False),
    ("5 HSV        ", check_hsv,                       False),
    ("6 Tests      ", check_offline_tests,             True),
]


def main():
    offline = "--offline" in sys.argv
    fails = 0
    for name, fn, offline_safe in CHECKS:
        if offline and not offline_safe:
            print(f"  [SKIP]  {name}  (offline mode)"); continue
        print(f"  ...     {name}  running", end="\r")
        try:
            ok, msg = fn()
        except Exception as ex:
            ok, msg = False, f"exception: {ex}"
        print(f"  [{_mark(ok)}] {name}  {msg}")
        if not ok:
            fails += 1
    print()
    if fails == 0:
        print(ANSI_GREEN + "PREFLIGHT OK — cleared for lab work." + ANSI_RESET)
    else:
        print(ANSI_RED + f"PREFLIGHT FAIL — {fails} check(s) red." +
              ANSI_RESET)
    return fails


if __name__ == "__main__":
    sys.exit(main())
