"""
Step-based fruit-sorting FSM for Simulink (time.time-free).

This is the Simulink-facing twin of sorting_controller.py. The original
controller runs a blocking loop keyed off `time.time()`, which cannot be
driven by a Simulink fixed-step solver. This version exposes a single
step(joints_cur, dt) call that a MATLAB Function block can invoke once
per solver step — state transitions and trajectory interpolation are
driven by accumulated simulation time.
"""

import numpy as np

from qarm_kinematics import forward_kinematics, inverse_kinematics
from trajectory import quintic_trajectory


# State IDs (ints so Simulink can log them cleanly)
S_INIT          = 0
S_GO_HOME       = 1
S_SELECT        = 2
S_APPROACH      = 3
S_DESCEND       = 4
S_CLOSE_GRIP    = 5
S_ASCEND_PICK   = 6
S_MOVE_BASKET   = 7
S_DESCEND_PLACE = 8
S_OPEN_GRIP     = 9
S_ASCEND_PLACE  = 10
S_DONE          = 11

STATE_NAMES = {
    S_INIT: 'INIT', S_GO_HOME: 'GO_HOME', S_SELECT: 'SELECT',
    S_APPROACH: 'APPROACH', S_DESCEND: 'DESCEND', S_CLOSE_GRIP: 'CLOSE_GRIP',
    S_ASCEND_PICK: 'ASCEND_PICK', S_MOVE_BASKET: 'MOVE_BASKET',
    S_DESCEND_PLACE: 'DESCEND_PLACE', S_OPEN_GRIP: 'OPEN_GRIP',
    S_ASCEND_PLACE: 'ASCEND_PLACE', S_DONE: 'DONE',
}


class StepController:
    """Deterministic, dt-driven fruit-sorting FSM for Simulink."""

    BASKETS = {
        'strawberry': np.array([0.30, -0.20, 0.05]),
        'banana':     np.array([-0.30, -0.20, 0.05]),
        'tomato':     np.array([0.00, -0.35, 0.05]),
    }

    HOME_POS    = np.array([0.45, 0.00, 0.49])
    SAFE_Z      = 0.20
    APPROACH_Z  = 0.15
    PICK_Z      = 0.02    # absolute floor — never command below this
    PICK_OFFSET = 0.02    # descend to (fruit_z - PICK_OFFSET) clamped to PICK_Z
    PLACE_Z     = 0.10

    T_TRANSIT  = 2.0
    T_APPROACH = 1.0
    T_PICK     = 0.8
    T_DWELL    = 0.5

    # Joint limits used for pre-flight reachability. Matches QArmDriver.
    JOINT_LIMITS = np.array([
        [-2.967, 2.967], [-1.484, 1.484],
        [-1.658, 1.309], [-2.793, 2.793],
    ])

    def __init__(self, fruit_positions=None, fruit_types=None):
        self.state = S_GO_HOME
        self.t = 0.0
        self.sorted_count = 0

        self.fruit_queue = []
        if fruit_positions is not None and fruit_types is not None:
            for p, typ in zip(fruit_positions, fruit_types):
                self.fruit_queue.append({'pos': np.asarray(p, dtype=float),
                                         'type': str(typ)})

        # Active trajectory (Cartesian)
        self._traj_start_t = 0.0
        self._traj_T       = 0.0
        self._p0           = np.zeros(3)
        self._p1           = np.zeros(3)
        self._dwell_start  = 0.0
        self._last_ee      = self.HOME_POS.copy()
        self._last_gripper = 0.0
        self._started      = False

    # ------------------------------------------------------------------
    def load_single_pick(self, base_xyz, fruit_type):
        """
        Queue a single target for one pick-place cycle, and reset FSM.

        Parity with FruitSortingController.pick_single for sim callers
        that drive step() themselves (e.g. Simulink facade, test harness).
        The caller is responsible for pumping step() until the returned
        `done` flag is True; this helper only sets up state.

        Parameters
        ----------
        base_xyz : array_like, shape (3,)
        fruit_type : str  — one of 'banana' | 'tomato' | 'strawberry'.
        """
        self.fruit_queue = [{
            'pos': np.asarray(base_xyz, dtype=float),
            'type': str(fruit_type),
        }]
        self.state = S_GO_HOME
        self.t = 0.0
        self.sorted_count = 0
        self._started = False
        self._last_ee = self.HOME_POS.copy()
        self._last_gripper = 0.0

    # ------------------------------------------------------------------
    def step(self, joints_cur, dt):
        """Advance the FSM one solver tick.

        Parameters
        ----------
        joints_cur : array_like of length 4, current joint angles (rad)
        dt         : float, time increment since previous call (s)

        Returns
        -------
        (phi_cmd[4], gripper_cmd, state_id, done)
        """
        self.t += float(dt)
        joints_cur = np.asarray(joints_cur, dtype=float).reshape(4)
        ee_pos, _ = forward_kinematics(joints_cur)

        if not self._started:
            self._start_move(ee_pos, self.HOME_POS, self.T_TRANSIT)
            self._started = True
            self.state = S_GO_HOME

        if self.state == S_GO_HOME:
            if self._track(ee_pos, 0.0):
                if not self.fruit_queue:
                    self.state = S_DONE
                else:
                    self.state = S_SELECT

        elif self.state == S_SELECT:
            target = self.fruit_queue.pop(0)
            self._current = target
            self._basket = self.BASKETS.get(target['type'], self.BASKETS['tomato'])
            approach = target['pos'].copy(); approach[2] = self.APPROACH_Z
            pick_chk = target['pos'].copy()
            pick_chk[2] = self._compute_pick_z(target['pos'])
            # Pre-flight reachability: if either approach or pick fails IK
            # or hits a joint limit, skip to next fruit (or DONE).
            if (self._ik_safe(approach) is None or
                    self._ik_safe(pick_chk) is None):
                self._current = None
                if self.fruit_queue:
                    self._start_move(ee_pos, self.HOME_POS, self.T_TRANSIT)
                    self.state = S_GO_HOME
                else:
                    self.state = S_DONE
            else:
                self._start_move(ee_pos, approach, self.T_TRANSIT)
                self.state = S_APPROACH

        elif self.state == S_APPROACH:
            if self._track(ee_pos, 0.0):
                pick = self._current['pos'].copy()
                pick[2] = self._compute_pick_z(self._current['pos'])
                self._start_move(ee_pos, pick, self.T_PICK)
                self.state = S_DESCEND

        elif self.state == S_DESCEND:
            if self._track(ee_pos, 0.0):
                self._dwell_start = self.t
                self.state = S_CLOSE_GRIP

        elif self.state == S_CLOSE_GRIP:
            self._hold(ee_pos, 1.0)
            if self.t - self._dwell_start >= self.T_DWELL:
                ascend = ee_pos.copy(); ascend[2] = self.SAFE_Z
                self._start_move(ee_pos, ascend, self.T_APPROACH)
                self.state = S_ASCEND_PICK

        elif self.state == S_ASCEND_PICK:
            if self._track(ee_pos, 1.0):
                above = self._basket.copy(); above[2] = self.SAFE_Z
                self._start_move(ee_pos, above, self.T_TRANSIT)
                self.state = S_MOVE_BASKET

        elif self.state == S_MOVE_BASKET:
            if self._track(ee_pos, 1.0):
                place = self._basket.copy(); place[2] = self.PLACE_Z
                self._start_move(ee_pos, place, self.T_APPROACH)
                self.state = S_DESCEND_PLACE

        elif self.state == S_DESCEND_PLACE:
            if self._track(ee_pos, 1.0):
                self._dwell_start = self.t
                self.state = S_OPEN_GRIP

        elif self.state == S_OPEN_GRIP:
            self._hold(ee_pos, 0.0)
            if self.t - self._dwell_start >= self.T_DWELL:
                self.sorted_count += 1
                ascend = ee_pos.copy(); ascend[2] = self.SAFE_Z
                self._start_move(ee_pos, ascend, self.T_APPROACH)
                self.state = S_ASCEND_PLACE

        elif self.state == S_ASCEND_PLACE:
            if self._track(ee_pos, 0.0):
                self._start_move(ee_pos, self.HOME_POS, self.T_TRANSIT)
                self.state = S_GO_HOME

        elif self.state == S_DONE:
            self._hold(self.HOME_POS, 0.0)

        # Guarded IK — on failure, hold the last commanded joints so a
        # Simulink solver step doesn't raise into the model runner.
        phi = self._ik_safe(self._last_ee)
        if phi is None:
            phi = getattr(self, "_last_phi", np.zeros(4))
        self._last_phi = np.asarray(phi, dtype=float).copy()
        done = 1 if self.state == S_DONE else 0
        return phi, float(self._last_gripper), int(self.state), int(done)

    # ------------------------------------------------------------------
    def _start_move(self, p_start, p_end, T):
        self._traj_start_t = self.t
        self._traj_T = max(float(T), 1e-6)
        self._p0 = np.asarray(p_start, dtype=float).copy()
        self._p1 = np.asarray(p_end, dtype=float).copy()

    def _track(self, ee_pos, gripper):
        elapsed = self.t - self._traj_start_t
        if elapsed >= self._traj_T:
            self._last_ee = self._p1.copy()
            self._last_gripper = float(gripper)
            return True
        p = quintic_trajectory(self._p0, self._p1, self._traj_T, elapsed)
        self._last_ee = p
        self._last_gripper = float(gripper)
        return False

    def _hold(self, ee_pos, gripper):
        self._last_ee = np.asarray(ee_pos, dtype=float).copy()
        self._last_gripper = float(gripper)

    # ------------------------------------------------------------------
    def _compute_pick_z(self, fruit_xyz):
        """Z to descend to for a pick. Uses detected fruit_z minus an
        offset so the gripper straddles the fruit, clamped to the
        PICK_Z absolute floor so we never drive into the table."""
        return max(float(self.PICK_Z), float(fruit_xyz[2]) - self.PICK_OFFSET)

    def _ik_safe(self, target_pos):
        """Return IK joints for target_pos or None if unreachable / over
        joint limit. Used for pre-flight reachability checks and to guard
        the per-step IK call so Simulink solver steps don't raise."""
        try:
            phi = inverse_kinematics(target_pos, gamma=0.0)
        except Exception:
            return None
        phi = np.asarray(phi, dtype=float)
        for i in range(4):
            lo, hi = self.JOINT_LIMITS[i]
            if phi[i] < lo or phi[i] > hi:
                return None
        return phi
