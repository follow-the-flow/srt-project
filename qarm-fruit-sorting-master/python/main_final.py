"""
Top-level launcher for the autonomous fruit-sorting pipeline.

Loads session_cal.json, connects the QArm + D415, and hands off to
picker_viewer.run_picker_loop for the category-batch picker UI.

See docs/superpowers/specs/2026-04-22-overhead-vision-pipeline-design.md §4.1.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from session_cal import SessionCal

SESSION_CAL_PATH = os.path.join(os.path.dirname(_HERE), "session_cal.json")
STALE_HOURS = 12


def _load_session_cal(path):
    if not os.path.exists(path):
        sys.exit(f"[abort] {path} missing — run calibrate_chessboard.py")
    cal = SessionCal.load(path)
    try:
        age = _dt.datetime.now() - _dt.datetime.fromisoformat(cal.timestamp)
        age_h = age.total_seconds() / 3600.0
        if age_h > STALE_HOURS:
            print(f"[warn] session_cal is {age_h:.1f} h old "
                  f"(threshold {STALE_HOURS} h) — consider recalibrating")
    except Exception:
        pass
    return cal


def main():
    parser = argparse.ArgumentParser(
        description="QArm autonomous fruit-sorting picker")
    parser.add_argument("--dry-run", action="store_true",
        help="Print plan and exit without moving the arm.")
    args = parser.parse_args()

    print("=" * 60)
    print("  QArm Fruit Sorting — Autonomous Picker")
    print("=" * 60)
    print(f"  session_cal  : {SESSION_CAL_PATH}")
    cal = _load_session_cal(SESSION_CAL_PATH)
    print(f"  cam height   : {cal.camera_height_above_table_m:.3f} m")
    print(f"  chess origin : {cal.chess_origin_in_base_m.round(3)}")
    print(f"  homography   : RMS {cal.homography_reproj_rms_px:.2f} px")

    if args.dry_run:
        print("\n[dry-run] session_cal OK. Skipping hardware + picker.")
        return 0

    # Connect hardware
    from qarm_driver import QArmDriver
    from camera import QArmCamera
    from sorting_controller import FruitSortingController
    from picker_viewer import run_picker_loop

    driver = QArmDriver()
    driver.connect()
    camera = QArmCamera()
    camera.open()
    controller = FruitSortingController(driver)
    try:
        run_picker_loop(driver, camera, cal, controller)
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C — exiting")
    finally:
        try: camera.close()
        except Exception: pass
        try: driver.home(duration=3.0)
        except Exception: pass
        try: driver.disconnect()
        except Exception: pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
