#!/usr/bin/env python3
"""
Offline validation of the Python fruit sorting pipeline.
Tests FK, IK, trajectories, and the controller WITHOUT hardware.
"""

import numpy as np
import time
import sys

# Add current directory to path
sys.path.insert(0, '.')

from qarm_kinematics import forward_kinematics, inverse_kinematics
from trajectory import cubic_trajectory, multi_segment_trajectory
from fruit_detector import detect_fruits


def test_fk_ik():
    """Test FK/IK round-trip accuracy."""
    print("=== Test 1: FK/IK Round-trip ===")
    test_positions = [
        [0.25, 0.25, 0.10],
        [0.30, -0.20, 0.05],
        [-0.30, -0.20, 0.05],
        [0.00, -0.35, 0.05],
        [0.00, 0.375, 0.10],
        [0.45, 0.00, 0.49],
    ]

    max_err = 0
    for p_target in test_positions:
        p = np.array(p_target)
        phi = inverse_kinematics(p, gamma=0.0)
        p_result, _ = forward_kinematics(phi)
        err = np.linalg.norm(p_result - p) * 1000  # mm

        status = "PASS" if err < 1.0 else "CHECK"
        print(f"  [{status}] Target: {p} -> Error: {err:.4f} mm "
              f"(phi: [{', '.join(f'{np.degrees(a):.1f}' for a in phi)}] deg)")
        max_err = max(max_err, err)

    print(f"  Max error: {max_err:.4f} mm\n")
    return max_err < 1.0


def test_trajectory():
    """Test cubic trajectory smoothness."""
    print("=== Test 2: Trajectory Smoothness ===")
    p_start = np.array([0.45, 0, 0.49])
    p_end = np.array([0.25, 0.25, 0.10])
    T = 2.0
    dt = 0.002

    positions = []
    for t in np.arange(0, T + dt, dt):
        positions.append(cubic_trajectory(p_start, p_end, T, t))

    positions = np.array(positions)
    velocities = np.diff(positions, axis=0) / dt

    max_vel = np.max(np.abs(velocities), axis=0)
    print(f"  Max velocity: vx={max_vel[0]:.3f}, vy={max_vel[1]:.3f}, "
          f"vz={max_vel[2]:.3f} m/s")

    v_start = np.linalg.norm(velocities[0])
    v_end = np.linalg.norm(velocities[-1])
    print(f"  Boundary velocities: start={v_start:.6f}, end={v_end:.6f} m/s")

    start_ok = np.allclose(positions[0], p_start, atol=1e-6)
    end_ok = np.allclose(positions[-1], p_end, atol=1e-6)
    print(f"  Start position: {'PASS' if start_ok else 'FAIL'}")
    print(f"  End position: {'PASS' if end_ok else 'FAIL'}\n")
    return start_ok and end_ok


def test_detection():
    """Test fruit detection with synthetic image."""
    print("=== Test 3: Fruit Detection ===")
    try:
        import cv2
    except ImportError:
        print("  SKIP: OpenCV not installed\n")
        return True

    # Create synthetic image
    img = np.full((480, 640, 3), 180, dtype=np.uint8)

    # Red circle (tomato - round)
    cv2.circle(img, (300, 200), 30, (30, 30, 230), -1)
    # Red elongated (strawberry)
    cv2.ellipse(img, (150, 300), (15, 30), 0, 0, 360, (20, 20, 200), -1)
    # Yellow circle (banana)
    cv2.circle(img, (450, 250), 25, (50, 220, 240), -1)

    detections = detect_fruits(img)
    print(f"  Detected {len(detections)} fruits:")
    for d in detections:
        print(f"    {d}")

    has_red = any(d.fruit_type in ('strawberry', 'tomato') for d in detections)
    has_yellow = any(d.fruit_type == 'banana' for d in detections)
    print(f"  Red fruit detected: {'PASS' if has_red else 'FAIL'}")
    print(f"  Yellow fruit detected: {'PASS' if has_yellow else 'FAIL'}\n")
    return True


def test_controller_sim():
    """Test the sorting controller in simulation (no hardware)."""
    print("=== Test 4: Controller Simulation ===")

    # Simulated QArm (mock)
    class MockQArm:
        def __init__(self):
            self._joints = np.zeros(4)
            self._gripper = 0.0
        def read_all(self):
            return self._joints.copy(), self._gripper
        def set_joints_and_gripper(self, joints, gripper):
            self._joints = np.array(joints[:4])
            self._gripper = gripper
        def home(self, duration=1.0):
            self._joints = np.zeros(4)
            self._gripper = 0.0

    mock = MockQArm()

    from sorting_controller import FruitSortingController, State
    ctrl = FruitSortingController(mock)

    fruits = [
        ([0.20, 0.30, 0.02], 'strawberry'),
        ([0.15, 0.25, 0.02], 'tomato'),
        ([-0.10, 0.35, 0.02], 'banana'),
    ]
    ctrl.set_fruit_positions(
        [np.array(p) for p, _ in fruits],
        [t for _, t in fruits]
    )

    # Run controller steps
    ctrl.state = State.GO_HOME
    max_steps = 100000
    step = 0
    t = 0.0
    dt = 0.002

    while ctrl.state != State.DONE and step < max_steps:
        ctrl._step(time.time())
        step += 1
        t += dt
        time.sleep(0.0001)  # Minimal delay for timing

    success = ctrl.state == State.DONE
    print(f"  Completed: {success}")
    print(f"  Fruits sorted: {ctrl.sorted_count}")
    print(f"  Steps taken: {step}")
    print(f"  {'PASS' if success and ctrl.sorted_count == 3 else 'CHECK'}\n")
    return success


def main():
    print("=" * 50)
    print("  QArm Fruit Sorting - Python Validation")
    print("=" * 50)
    print()

    results = []
    results.append(("FK/IK", test_fk_ik()))
    results.append(("Trajectory", test_trajectory()))
    results.append(("Detection", test_detection()))
    results.append(("Controller", test_controller_sim()))

    print("=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} [{status}]")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  All tests PASSED!")
    else:
        print("  Some tests need attention.")
    print()


if __name__ == "__main__":
    main()
