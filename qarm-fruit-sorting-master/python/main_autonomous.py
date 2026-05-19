#!/usr/bin/env python3
"""
Main script: Autonomous Fruit Sorting with QArm + Quanser SDK.

Usage:
    python main_autonomous.py

Requires:
    - QArm connected via USB
    - Intel RealSense D415 camera (optional, can use manual positions)
    - Quanser SDK Python API installed
    - numpy, opencv-python
"""

import sys
import numpy as np
import time

from qarm_driver import QArmDriver
from qarm_kinematics import forward_kinematics, inverse_kinematics
from sorting_controller import FruitSortingController
from camera import QArmCamera
from fruit_detector import detect_fruits

# ============================================================
# CONFIGURATION -- ADJUST THESE TO YOUR SETUP
# ============================================================

# Basket positions (metres, in QArm base frame)
BASKET_STRAWBERRY = [0.30, -0.20, 0.05]
BASKET_BANANA = [-0.30, -0.20, 0.05]
BASKET_TOMATO = [0.00, -0.35, 0.05]

# Camera calibration: camera-to-base transform
# REPLACE with your actual calibration from calibrate_camera.py
T_CAM_TO_BASE = np.eye(4)

# Set to True to use camera for detection, False for manual positions
USE_CAMERA = False

# Manual fruit positions (if USE_CAMERA = False)
# Format: list of (position, type) tuples
MANUAL_FRUITS = [
    ([0.20, 0.30, 0.02], 'strawberry'),
    ([0.15, 0.25, 0.02], 'tomato'),
    ([-0.10, 0.35, 0.02], 'banana'),
]


def main():
    print("=" * 60)
    print("  QArm Fruit Sorting - Autonomous Mode")
    print("  Quanser SDK for Windows")
    print("=" * 60)

    # --- Connect to QArm ---
    qarm = QArmDriver()
    try:
        qarm.connect()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Make sure the QArm is connected via USB and powered on.")
        print("If testing without hardware, use validate_python.py instead.")
        sys.exit(1)

    try:
        # --- Move to home ---
        print("\nMoving to home position...")
        qarm.home(duration=3.0)
        time.sleep(0.5)

        # --- Set up controller ---
        controller = FruitSortingController(qarm)

        # Update basket positions
        controller.BASKETS['strawberry'] = np.array(BASKET_STRAWBERRY)
        controller.BASKETS['banana'] = np.array(BASKET_BANANA)
        controller.BASKETS['tomato'] = np.array(BASKET_TOMATO)

        # --- Detect fruits ---
        if USE_CAMERA:
            print("\nOpening camera for fruit detection...")
            cam = QArmCamera()
            cam.open()
            time.sleep(1.0)  # Let camera warm up

            color, depth = cam.read()
            detections = detect_fruits(color, depth)
            cam.close()

            if not detections:
                print("No fruits detected! Check lighting and HSV thresholds.")
                return

            positions = []
            types = []
            for det in detections:
                row, col = det.centroid
                d = depth[row, col] if depth is not None else 300
                p_world = QArmCamera().pixel_to_world(row, col, d, T_CAM_TO_BASE)
                positions.append(p_world)
                types.append(det.fruit_type)
                print(f"  Detected: {det.fruit_type} at {p_world}")

            controller.set_fruit_positions(positions, types)
        else:
            print("\nUsing manual fruit positions...")
            positions = [np.array(p) for p, _ in MANUAL_FRUITS]
            types = [t for _, t in MANUAL_FRUITS]
            controller.set_fruit_positions(positions, types)

        # --- Run sorting ---
        print("\nStarting autonomous sorting...")
        input("Press ENTER to begin (ensure workspace is clear)...")

        controller.run_autonomous(dt=0.002)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    finally:
        print("\nReturning to home and disconnecting...")
        qarm.home(duration=3.0)
        qarm.disconnect()


if __name__ == "__main__":
    main()
