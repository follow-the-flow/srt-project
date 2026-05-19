"""
End-to-end integration tests against mock hardware.

Exercises the FSM and controller pipeline without a real QArm so
regressions are caught before lab sessions. Covers:

1. sorting_controller.FruitSortingController FSM (blocking loop) against
   a MockQArm. Verifies full pick/place cycle, IK safety on unreachable
   fruits, gripper ramp + readback, and per-fruit Z adaptation.
2. sorting_controller_sim.StepController (Simulink-facing twin) against
   MockQArm. Verifies the stepwise API completes without exceptions for
   reachable fruits.
3. Trajectory and kinematic utilities (quintic, jog clamping, etc.).

Vision tests (fruit detection) are now in test_fruit_detector.py.

Run with:
    C:/Python313/python.exe python/test_integration.py
Exit code is the number of failed sections (0 = all pass).
"""

import os
import sys
import time
import traceback
import numpy as np

# Allow running from repo root or python/ dir
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qarm_kinematics import forward_kinematics, inverse_kinematics
from sorting_controller import FruitSortingController, State, GRIP_OPEN


# ------------------------------------------------------------------------
# Mocks
# ------------------------------------------------------------------------
class MockQArm:
    """Record-only qarm: set_joints_and_gripper appends to a log, read_all
    returns the last commanded joints+grip. Tracks 'perfectly'."""

    JOINT_LIMITS = np.array([
        [-2.967, 2.967], [-1.484, 1.484],
        [-1.658, 1.309], [-2.793, 2.793],
    ])

    def __init__(self):
        self._joints = np.zeros(4, dtype=float)
        self._grip = 0.15
        self.cmd_log = []

    def read_all(self):
        return self._joints.copy(), float(self._grip)

    def set_joints_and_gripper(self, phi, g):
        phi = np.asarray(phi, dtype=float).flatten()[:4]
        self._joints = phi.copy()
        self._grip = float(g)
        self.cmd_log.append((phi.copy(), float(g)))


# ------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------
def test_controller_full_cycle_pick_only():
    """FruitSortingController pick_only mode: detect -> pick -> lift -> open."""
    name = "controller_pick_only"
    q = MockQArm()
    c = FruitSortingController(q, pick_only=True)
    # Compress timing so the test runs fast
    c.T_TRANSIT = 0.05
    c.T_APPROACH = 0.05
    c.T_PICK = 0.05
    c.T_DWELL = 0.02
    c.T_GRIP = 0.02
    c.T_SETTLE = 0.02
    # Two reachable fruits at distinct Z values
    positions = [np.array([0.35, 0.10, 0.13]),
                 np.array([0.30, -0.05, 0.08])]
    types = ["strawberry", "tomato"]
    c.set_fruit_positions(positions, types)
    c.state = State.GO_HOME

    t0 = time.time()
    steps = 0
    while c.state != State.DONE and steps < 10000:
        c._step(time.time())
        steps += 1
        time.sleep(0.001)
        if time.time() - t0 > 10.0:
            raise AssertionError("controller did not complete within 10s")

    assert c.sorted_count == 2, f"sorted {c.sorted_count}/2"
    assert len(q.cmd_log) > 100, f"too few commands: {len(q.cmd_log)}"
    # Gripper should sweep through intermediate values, not snap
    grips = np.array([g for _, g in q.cmd_log])
    unique = np.unique(np.round(grips, 2))
    assert len(unique) >= 5, \
        f"gripper snapped (only {len(unique)} distinct values): {unique}"
    return name, True, (f"2/2 sorted in {steps} steps, "
                        f"{len(unique)} distinct grip values")


def test_controller_skips_unreachable():
    """Unreachable fruit must be skipped via _ik_safe pre-flight, not crash."""
    name = "controller_skips_unreachable"
    q = MockQArm()
    c = FruitSortingController(q, pick_only=True)
    c.T_TRANSIT = 0.05; c.T_APPROACH = 0.05; c.T_PICK = 0.05
    c.T_DWELL = 0.02; c.T_GRIP = 0.02; c.T_SETTLE = 0.02
    # First fruit is far outside workspace (>1 m), second is reachable
    c.set_fruit_positions(
        [np.array([2.0, 0.0, 0.5]), np.array([0.35, 0.1, 0.13])],
        ["strawberry", "tomato"])
    c.state = State.GO_HOME

    t0 = time.time()
    steps = 0
    while c.state != State.DONE and steps < 10000:
        c._step(time.time())
        steps += 1
        time.sleep(0.001)
        if time.time() - t0 > 10.0:
            raise AssertionError("controller hung on unreachable fruit")

    # Only the reachable one counts
    assert c.sorted_count == 1, f"expected 1 sort, got {c.sorted_count}"
    return name, True, "unreachable skipped, reachable sorted"


def test_controller_full_sort_mode():
    """Non-pick-only mode: full approach/pick/transit/place/release cycle."""
    name = "controller_full_sort"
    q = MockQArm()
    c = FruitSortingController(q, pick_only=False)
    c.T_TRANSIT = 0.05; c.T_APPROACH = 0.05; c.T_PICK = 0.05
    c.T_DWELL = 0.02; c.T_GRIP = 0.02; c.T_SETTLE = 0.02
    # Override baskets to reachable positions
    c.BASKETS = {
        'strawberry': np.array([0.30, -0.15, 0.10]),
        'banana':     np.array([0.30, -0.20, 0.10]),
        'tomato':     np.array([0.25, -0.25, 0.10]),
    }
    c.set_fruit_positions(
        [np.array([0.35, 0.10, 0.13]), np.array([0.30, -0.05, 0.08])],
        ["strawberry", "tomato"])
    c.state = State.GO_HOME

    t0 = time.time()
    steps = 0
    states_seen = set()
    while c.state != State.DONE and steps < 20000:
        states_seen.add(c.state)
        c._step(time.time())
        steps += 1
        time.sleep(0.001)
        if time.time() - t0 > 15.0:
            raise AssertionError("full-sort cycle hung")

    assert c.sorted_count == 2, f"sorted {c.sorted_count}/2"
    expected_states = {State.GO_HOME, State.SELECT_FRUIT, State.APPROACH,
                       State.DESCEND, State.CLOSE_GRIPPER, State.ASCEND_PICK,
                       State.MOVE_TO_BASKET, State.DESCEND_PLACE,
                       State.OPEN_GRIPPER, State.ASCEND_PLACE}
    missing = expected_states - states_seen
    assert not missing, f"states not visited: {[s.name for s in missing]}"
    return name, True, f"2/2 sorted; all 10 transit states visited"


def test_quintic_boundary_conditions():
    """Quintic must have zero velocity AND zero acceleration at both ends
    (jerk-continuous). Numerical check via finite differences."""
    name = "quintic_boundary_conditions"
    from trajectory import quintic_trajectory
    T = 1.0
    ps = np.array([0.0, 0.0, 0.0])
    pe = np.array([1.0, -0.5, 0.3])
    h = 1e-4

    def p(t):
        return quintic_trajectory(ps, pe, T, t)

    # Velocity at endpoints (forward / backward differences)
    v0 = (p(h) - p(0)) / h
    vT = (p(T) - p(T - h)) / h
    # Acceleration at endpoints (central diff near the edge)
    a0 = (p(2 * h) - 2 * p(h) + p(0)) / (h * h)
    aT = (p(T) - 2 * p(T - h) + p(T - 2 * h)) / (h * h)

    assert np.linalg.norm(v0) < 1e-3, f"v(0) not zero: {v0}"
    assert np.linalg.norm(vT) < 1e-3, f"v(T) not zero: {vT}"
    assert np.linalg.norm(a0) < 1e-1, f"a(0) not zero: {a0}"
    assert np.linalg.norm(aT) < 1e-1, f"a(T) not zero: {aT}"

    # Endpoint reached exactly
    err = np.linalg.norm(p(T) - pe)
    assert err < 1e-9, f"p(T) != pe, err={err}"

    # Mid-point sanity (smoothstep-like should be at 0.5 * (ps+pe))
    mid = p(0.5 * T)
    expected_mid = 0.5 * (ps + pe)
    assert np.linalg.norm(mid - expected_mid) < 1e-9, \
        f"mid not halfway: got {mid}, expected {expected_mid}"
    return name, True, "v=0, a=0 at both endpoints; midpoint exact"


def test_sim_controller():
    """StepController (Simulink-facing twin) runs without raising."""
    name = "sim_controller_step_api"
    from sorting_controller_sim import StepController, S_DONE

    positions = [np.array([0.35, 0.10, 0.13]),
                 np.array([0.30, -0.05, 0.10])]
    types = ["strawberry", "tomato"]

    sc = StepController(fruit_positions=positions, fruit_types=types)
    # Shorten timings
    sc.T_TRANSIT = 0.2
    sc.T_APPROACH = 0.1
    sc.T_PICK = 0.1
    sc.T_DWELL = 0.05

    dt = 0.01
    # Start at home joints
    joints = np.zeros(4, dtype=float)
    for _ in range(5000):
        phi, grip, state_id, done = sc.step(joints, dt)
        # Simulate perfect tracking
        joints = np.asarray(phi, dtype=float).reshape(4)
        if done:
            break
    else:
        raise AssertionError("StepController did not finish in 5000 steps")

    assert sc.state == S_DONE, f"final state {sc.state} != DONE ({S_DONE})"
    assert sc.sorted_count == 2, f"sorted {sc.sorted_count}/2"
    return name, True, f"2/2 sorted in step API"


def test_sim_controller_skips_unreachable():
    """StepController must pre-flight IK and skip unreachable fruits
    without raising out of the step() call (Simulink needs this)."""
    name = "sim_controller_skips_unreachable"
    from sorting_controller_sim import StepController, S_DONE

    # First is far outside workspace, second is reachable
    positions = [np.array([2.5, 0.0, 0.5]),
                 np.array([0.35, 0.10, 0.13])]
    types = ["strawberry", "tomato"]
    sc = StepController(fruit_positions=positions, fruit_types=types)
    sc.T_TRANSIT = 0.2; sc.T_APPROACH = 0.1; sc.T_PICK = 0.1; sc.T_DWELL = 0.05

    dt = 0.01
    joints = np.zeros(4, dtype=float)
    for _ in range(8000):
        phi, grip, state_id, done = sc.step(joints, dt)
        joints = np.asarray(phi, dtype=float).reshape(4)
        if done:
            break
    else:
        raise AssertionError("StepController hung on unreachable fruit")

    assert sc.state == S_DONE, f"final state {sc.state} != DONE"
    assert sc.sorted_count == 1, \
        f"expected 1 sort (reachable only), got {sc.sorted_count}"
    return name, True, "unreachable skipped, reachable sorted"


def test_sim_controller_uses_detected_z():
    """_compute_pick_z must use the detected fruit Z instead of PICK_Z when
    the detected value is above PICK_Z."""
    name = "sim_controller_pick_z"
    from sorting_controller_sim import StepController
    sc = StepController()
    # Floor behavior: detected 0.01 below the floor -> clamp to PICK_Z
    z_low = sc._compute_pick_z(np.array([0.3, 0.1, 0.005]))
    assert z_low == sc.PICK_Z, f"low fruit not clamped: {z_low}"
    # Above floor: use fruit_z - offset
    z_high = sc._compute_pick_z(np.array([0.3, 0.1, 0.15]))
    expected = 0.15 - sc.PICK_OFFSET
    assert abs(z_high - expected) < 1e-9, f"expected {expected}, got {z_high}"
    return name, True, "pick_z clamps floor and subtracts offset"


def test_set_gripper_ramp_extracted():
    """The shared helper ramps from `from` to `target` over `duration`
    seconds using smoothstep, and after the ramp exposes `held_grip`
    matching the settled value reported by the mocked qarm."""
    name = "set_gripper_ramp_helper"
    from sorting_controller import FruitSortingController, GRIP_OPEN, GRIP_CLOSE
    q = MockQArm()
    q.fake_grip = GRIP_OPEN  # mock tracks perfectly
    c = FruitSortingController(q)
    c.T_GRIP = 0.05  # quick
    # Precondition: helper exists on the controller class
    assert hasattr(c, "set_gripper_ramp"), "set_gripper_ramp missing"
    # Ramp from current to CLOSE
    result = c.set_gripper_ramp(GRIP_CLOSE)
    # Should have sent intermediate values, not just a single snap
    cmds = [g for _, g in q.cmd_log]
    unique = set(round(v, 2) for v in cmds)
    assert len(unique) >= 5, f"no ramp, only {unique}"
    # After settling, held_grip matches the mocked actual (which tracked)
    assert abs(c._held_grip - GRIP_CLOSE) < 0.05, \
        f"held_grip {c._held_grip} != {GRIP_CLOSE}"
    return name, True, f"ramp over {len(unique)} values, held_grip correct"


def test_sorting_controller_emits_trace_events():
    """When a logger is passed to FruitSortingController, it should emit
    PICK_ATTEMPT, GRIPPER_READBACK, and SORT_COMPLETE events during a
    pick-only run."""
    name = "controller_trace_events"
    from sorting_controller import FruitSortingController, State
    from trace_logger import TraceLogger
    import tempfile, os
    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, "test.log")
    logger = TraceLogger(log_path)

    q = MockQArm()
    c = FruitSortingController(q, pick_only=True, logger=logger)
    c.T_TRANSIT = 0.02; c.T_APPROACH = 0.02; c.T_PICK = 0.02
    c.T_DWELL = 0.01; c.T_GRIP = 0.01; c.T_SETTLE = 0.01
    c.set_fruit_positions([[0.35, 0.10, 0.13]], ["strawberry"])
    c.state = State.GO_HOME
    for _ in range(3000):
        c._step(time.time())
        time.sleep(0.001)
        if c.state == State.DONE:
            # Run one extra step so the DONE event fires.
            c._step(time.time())
            break
    logger.close()

    with open(log_path) as f:
        log = f.read()
    assert "PICK_ATTEMPT" in log, "no PICK_ATTEMPT event"
    assert "GRIPPER_READBACK" in log, "no GRIPPER_READBACK event"
    assert "SORT_COMPLETE" in log, "no SORT_COMPLETE event"
    return name, True, "3/3 trace events emitted"


def test_remote_jog_clamps_workspace():
    """Cartesian nudges that would leave the workspace box are refused."""
    name = "remote_jog_workspace_clamp"
    from remote_jog import cartesian_nudge, WORKSPACE_BOX
    # Near +X edge
    cur = np.array([WORKSPACE_BOX['x'][1] - 0.002, 0.0, 0.15])
    out = cartesian_nudge(cur, axis='x', step=0.01)  # would leave box
    assert out is None, f"expected clamp, got {out}"
    # Middle of box, small step, stays inside
    cur = np.array([0.30, 0.00, 0.15])
    out = cartesian_nudge(cur, axis='x', step=0.01)
    assert out is not None
    assert abs(out[0] - 0.31) < 1e-9
    return name, True, "clamps edge, accepts middle"


def test_remote_joint_jog_clamps_limits():
    name = "remote_joint_jog_clamp"
    from remote_jog import joint_nudge
    from qarm_driver import QArmDriver
    lims = QArmDriver.JOINT_LIMITS
    # Near upper limit on joint 1
    cur = np.array([lims[0, 1] - 0.001, 0, 0, 0], dtype=float)
    out = joint_nudge(cur, joint=0, step=np.deg2rad(5))
    assert out is None, "should clamp"
    # Middle of range
    cur = np.array([0.0, 0.0, 0.0, 0.0])
    out = joint_nudge(cur, joint=0, step=np.deg2rad(5))
    assert out is not None
    assert abs(out[0] - np.deg2rad(5)) < 1e-9
    return name, True, "clamps edge, accepts middle"


# ------------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------------
TESTS = [
    test_quintic_boundary_conditions,
    test_controller_full_cycle_pick_only,
    test_controller_skips_unreachable,
    test_controller_full_sort_mode,
    test_sim_controller,
    test_sim_controller_skips_unreachable,
    test_sim_controller_uses_detected_z,
    test_set_gripper_ramp_extracted,
    test_sorting_controller_emits_trace_events,
    test_remote_jog_clamps_workspace,
    test_remote_joint_jog_clamps_limits,
]


def main():
    print(f"Running {len(TESTS)} integration tests...\n")
    results = []
    for t in TESTS:
        try:
            name, ok, detail = t()
            results.append((name, ok, detail))
            print(f"  [OK]   {name}  — {detail}")
        except Exception as ex:
            name = t.__name__.replace("test_", "")
            results.append((name, False, str(ex)))
            print(f"  [FAIL] {name}  — {ex}")
            traceback.print_exc()
    fails = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed.")
    return fails


if __name__ == "__main__":
    sys.exit(main())
