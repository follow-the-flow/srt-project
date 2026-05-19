"""
Fruit Sorting Controller - State Machine for autonomous and remote operation.
Works with the Quanser SDK Python API.
"""

import numpy as np
import time
from enum import Enum, auto

from qarm_kinematics import forward_kinematics, inverse_kinematics
from trajectory import quintic_trajectory

GRIP_CLOSE = 0.90   # 2026-04-27 lab: briefly tried 0.72 (= 80%) to soften
                    # the strawberry grip, but reverted to 0.90 because
                    # the FSM already reads the settled jaw position into
                    # _held_grip after CLOSE_GRIPPER (so harder objects
                    # stall at their own thickness without sustained
                    # torque), and 0.72 left some strawberries gripped
                    # too loosely.
GRIP_OPEN = 0.15


class State(Enum):
    INIT = auto()
    GO_HOME = auto()
    SCAN = auto()
    SELECT_FRUIT = auto()
    APPROACH = auto()
    DESCEND = auto()
    CLOSE_GRIPPER = auto()
    ASCEND_PICK = auto()
    MOVE_TO_BASKET = auto()
    DESCEND_PLACE = auto()
    OPEN_GRIPPER = auto()
    ASCEND_PLACE = auto()
    DONE = auto()


class FruitSortingController:
    """
    Autonomous fruit sorting state machine.

    Sequences: HOME -> SCAN -> for each fruit:
        SELECT -> APPROACH -> DESCEND -> CLOSE -> ASCEND -> BASKET -> DESCEND -> OPEN -> ASCEND -> HOME
    """

    # Basket positions in QArm base frame (metres). Z is the **release**
    # height (gripper xyz when opening); x,y are the basket centre.
    # Banana taught 2026-04-28 by jogging the arm to the physical basket and
    # reading FK; strawberry/tomato share the same z so all three releases
    # happen at a uniform height. Re-teach via main_gui.py "teach basket"
    # if baskets are physically moved.
    BASKETS = {
        'strawberry': np.array([0.30,    -0.20,    0.0929]),
        'banana':     np.array([-0.2754, -0.1896,  0.0929]),
        'tomato':     np.array([0.00,    -0.35,    0.0929]),
    }

    # Key heights
    HOME_POS = np.array([0.45, 0.0, 0.49])
    SAFE_Z = 0.20
    APPROACH_Z = 0.15
    PICK_Z = 0.02          # absolute floor — never go below this
    PICK_OFFSET = 0.02     # subtract from fruit_z when computing pick height
                           # 2026-04-23 briefly tried 0.04 but FSM's Cartesian
                           # quintic stalls in singular joint space at low z;
                           # 0.02 was the value diag_pick_one used successfully.

    # Per-fruit-type pick-depth overrides. The default PICK_OFFSET +
    # PICK_Z apply unless overridden here. Strawberry descends a touch
    # deeper into the body so the gripper jaws cradle the wide section
    # rather than skimming the top of the soft skin and slipping off.
    # Tuning history (same lab session 2026-04-27):
    #   default  -> 0.022 m above table  : too shallow, gripper slipped
    #   {0.03, 0.015} -> 0.015 m         : too deep, gripper struck table
    #   {0.025, 0.018} -> 0.018 m        : current — 4 mm deeper than default
    PICK_DEPTH_OVERRIDES = {
        "strawberry": {"pick_offset": 0.025, "pick_z_floor": 0.018},
    }

    # Per-fruit-type close-gripper target overrides. Banana skin is firm
    # and the fruit is long, so it slipped at the global GRIP_CLOSE=0.90
    # in the 2026-04-28 lab — bumped to 1.00 (full close, clamped at the
    # max of the [0,1] gripper command range). The post-close readback
    # still pins _held_grip to the actual settled jaw position, so harder
    # fruits stall at their own thickness without sustained torque.
    GRIP_CLOSE_OVERRIDES = {
        "banana": 1.00,
    }

    # Timing
    T_TRANSIT = 2.0
    T_APPROACH = 1.5       # raised 2026-04-23 from 1.0
    T_PICK = 2.5           # raised 2026-04-23 from 0.8: 12 cm descent
                           # in 0.8 s saturated the arm's velocity and the
                           # FSM moved to CLOSE_GRIPPER while arm was still
                           # ~12 cm above target, closing on air. Matches
                           # diag_pick_one's slow_move_to_joints seconds=3.0.
    T_DWELL = 0.5
    T_GRIP = 0.8           # gripper open/close interpolation time
    T_SETTLE = 1.0         # added 2026-04-23: keep commanding target after
                           # interp ends so arm actually reaches it (arm has
                           # ~0.5-1.0 s tracking lag; without settle the FSM
                           # hands off to the next state while arm is still
                           # short of target, e.g. DESCEND -> CLOSE_GRIPPER
                           # while gripper is 10 cm above fruit).

    # Empirical x-offset applied at pick_single() time to compensate for the
    # residual base-frame projection bias observed 2026-04-24. Exposed as a
    # class attribute so it can be overridden per-session/per-rig without
    # touching FSM code, and so it is visible to anyone reading the contract.
    # Detection position is unchanged on screen — only the arm target shifts.
    PICK_BIAS_X = 0.05

    def __init__(self, qarm, camera=None, pick_only=False, logger=None):
        """
        Parameters
        ----------
        qarm : QArmDriver
            Connected QArm hardware driver.
        camera : QArmCamera, optional
            Camera for fruit detection.
        pick_only : bool
            If True, just pick up and lift each fruit (no basket placement).
        logger : TraceLogger, optional
            Event logger. When provided, the FSM emits PICK_ATTEMPT,
            GRIPPER_READBACK, and SORT_COMPLETE events for postprocessing
            by scripts/generate_report_plots.py.
        """
        self.qarm = qarm
        self.camera = camera
        self.pick_only = pick_only
        self.logger = logger
        self.state = State.INIT
        self.fruit_queue = []
        self.current_target = None
        self.target_basket = None
        self.sorted_count = 0
        self._traj_start = 0
        self._traj_duration = 0
        self._traj_start_pos = None
        self._traj_end_pos = None
        self._traj_start_joints = None   # joint-space interp (2026-04-23)
        self._traj_target_joints = None  # set by _start_move via IK
        self._driving = False            # reentrancy guard for _drive_until_done
        # Optional zero-arg callable invoked once per FSM tick from
        # _drive_until_done. Used by picker_viewer to render the live D415
        # feed during arm motion. Errors are caught + printed so display
        # issues never crash arm control.
        self.tick_observer = None
        # Gripper interpolation state — avoids -1289 overload stalls by
        # ramping the command and reading back the settled position after
        # the close, so we never keep torque against a stalled jaw.
        self._held_grip = GRIP_OPEN       # gripper value to send while transiting
        self._grip_from = GRIP_OPEN       # start of current grip interp
        self._grip_to = GRIP_OPEN         # end of current grip interp
        self._grip_start_time = 0.0       # wall clock start of current grip interp

    def set_fruit_positions(self, positions, types):
        """
        Manually set detected fruit positions (bypass camera).

        Parameters
        ----------
        positions : list of np.ndarray
            List of 3D positions in base frame.
        types : list of str
            List of fruit type strings.
        """
        self.fruit_queue = [
            {'pos': np.array(p), 'type': t}
            for p, t in zip(positions, types)
        ]
        print(f"Loaded {len(self.fruit_queue)} fruits into queue")

    def run_autonomous(self, dt=0.01):
        """
        Run the full autonomous sorting loop.

        Parameters
        ----------
        dt : float
            Control loop period (seconds). Default 100 Hz — plenty for
            trajectory tracking at QArm motion bandwidths.
        """
        print("\n=== AUTONOMOUS FRUIT SORTING ===")
        print(f"Fruits to sort: {len(self.fruit_queue)}")
        print(f"Baskets: {list(self.BASKETS.keys())}")
        print()

        self.state = State.GO_HOME
        start_time = time.time()
        self._drive_until_done(dt)
        total_time = time.time() - start_time
        print(f"\n=== SORTING COMPLETE ===")
        print(f"Sorted {self.sorted_count} fruits in {total_time:.1f} seconds")

    def _drive_until_done(self, dt):
        """Tight control loop. Pumps _step at `dt` cadence until state==DONE
        or an exception is caught. Shared between run_autonomous and
        pick_single. Not reentrant — guarded by `self._driving`."""
        if self._driving:
            raise RuntimeError(
                "FSM already being driven; run_autonomous / pick_single / "
                "set_gripper_ramp must not overlap")
        self._driving = True
        try:
            while self.state != State.DONE:
                loop_start = time.time()
                try:
                    self._step(time.time())
                except Exception as ex:
                    print(f"[ERROR] state {self.state.name}: {ex}")
                    print("[ERROR] aborting — arm held at last commanded pose")
                    break
                if self.tick_observer is not None:
                    try:
                        self.tick_observer()
                    except Exception as obs_ex:
                        print(f"[warn] tick_observer raised: {obs_ex}")
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self._driving = False

    def pick_single(self, base_xyz, fruit_type, dt=0.01):
        """
        Run ONE synchronous pick-place cycle for a single target.

        Parameters
        ----------
        base_xyz : array_like, shape (3,)
            XYZ in robot base frame (metres).
        fruit_type : str
            One of 'banana', 'tomato', 'strawberry' — selects the basket.
        dt : float
            Control loop period; same semantics as run_autonomous.

        Blocking; must not be called while another control method
        (`run_autonomous`, `set_gripper_ramp`, `pick_single`) is already
        driving the FSM. Resets state to GO_HOME on entry.

        Returns
        -------
        bool
            True if sorted_count increased (pick+place succeeded);
            False if the target was unreachable or an exception aborted
            the cycle.
        """
        prior = self.sorted_count
        pick_target = np.asarray(base_xyz, dtype=float).copy()
        pick_target[0] += self.PICK_BIAS_X
        self.fruit_queue = [{
            'pos': pick_target,
            'type': str(fruit_type),
        }]
        self.state = State.GO_HOME
        self._traj_end_pos = None
        self._done_logged = False
        self._ik_warn_fired = False
        self._drive_until_done(dt)
        return self.sorted_count > prior

    def _log(self, tag, **kv):
        """Forward an event to the attached TraceLogger if any."""
        if self.logger is not None:
            self.logger.log(tag, **kv)

    def _step(self, t):
        """Execute one step of the state machine."""
        joints, gripper = self.qarm.read_all()
        ee_pos, _ = forward_kinematics(joints)

        # One-shot SORT_COMPLETE when DONE is first entered.
        if self.state == State.DONE and not getattr(self, "_done_logged", False):
            self._log("SORT_COMPLETE", sorted=self.sorted_count)
            self._done_logged = True

        if self.state == State.GO_HOME:
            # Arm trajectory exactly once per entry into GO_HOME.
            if self._traj_end_pos is None or \
               not np.allclose(self._traj_end_pos, self.HOME_POS):
                self._start_move(ee_pos, self.HOME_POS, self.T_TRANSIT)
                self._print_state("Moving to home")
            if self._track_trajectory(t, gripper_val=GRIP_OPEN):
                self._traj_end_pos = None  # mark traj consumed
                if not self.fruit_queue:
                    self.state = State.DONE
                else:
                    self.state = State.SELECT_FRUIT

        elif self.state == State.INIT:
            # Legacy re-entry point (post-place). Route through GO_HOME so
            # the trajectory actually tracks.
            self._traj_end_pos = None
            self.state = State.GO_HOME

        elif self.state == State.SELECT_FRUIT:
            self.current_target = self.fruit_queue.pop(0)
            basket_name = self.current_target['type']
            self.target_basket = self.BASKETS.get(basket_name, self.BASKETS['tomato'])

            approach_pos = self.current_target['pos'].copy()
            approach_pos[2] = self.APPROACH_Z
            # Pre-validate IK on approach and pick positions — skip unreachable
            pick_pos_chk = self.current_target['pos'].copy()
            pick_pos_chk[2] = self._compute_pick_z(
                self.current_target['pos'],
                fruit_type=self.current_target['type'])
            if self._ik_safe(approach_pos) is None or \
               self._ik_safe(pick_pos_chk) is None:
                self._print_state(
                    f"[skip] {self.current_target['type']} at "
                    f"{self.current_target['pos'].round(3)} unreachable — "
                    f"next fruit")
                self.current_target = None
                # Go back home and try next fruit
                self._traj_end_pos = None
                self.state = State.GO_HOME
                return
            self._log("PICK_ATTEMPT",
                       type=self.current_target['type'],
                       pos=self.current_target['pos'])
            self._start_move(ee_pos, approach_pos, self.T_TRANSIT)
            self.state = State.APPROACH
            self._print_state(f"Approaching {self.current_target['type']}")

        elif self.state == State.APPROACH:
            if self._track_trajectory(t, gripper_val=GRIP_OPEN):
                pick_pos = self.current_target['pos'].copy()
                pick_pos[2] = self._compute_pick_z(
                    self.current_target['pos'],
                    fruit_type=self.current_target['type'])
                self._start_move(ee_pos, pick_pos, self.T_PICK)
                self.state = State.DESCEND
                self._print_state(f"Descending to pick (z={pick_pos[2]:.3f})")

        elif self.state == State.DESCEND:
            if self._track_trajectory(t, gripper_val=GRIP_OPEN):
                grip_close = self._grip_close_for(self.current_target['type'])
                self._start_grip_interp(grip_close, duration=self.T_GRIP)
                self.state = State.CLOSE_GRIPPER
                self._print_state(
                    f"Closing gripper to {grip_close:.2f} (ramped)")

        elif self.state == State.CLOSE_GRIPPER:
            # Ramp gripper smoothly from OPEN to CLOSE (avoids -1289 overload).
            g = self._update_grip_interp()
            self._execute_position(ee_pos, g)
            if self._grip_interp_done():
                grip_close = self._grip_close_for(self.current_target['type'])
                # Read the ACTUAL settled gripper position and hold THAT
                # value. If jaws stalled on the fruit, actual < target and
                # we stop fighting the motor — no residual torque.
                try:
                    _, actual = self.qarm.read_all()
                    self._held_grip = float(actual)
                except Exception:
                    self._held_grip = grip_close
                self._log("GRIPPER_READBACK",
                           type=self.current_target['type'],
                           target=grip_close,
                           actual=self._held_grip,
                           stall=(grip_close - self._held_grip))
                ascend_pos = ee_pos.copy()
                ascend_pos[2] = self.SAFE_Z
                self._start_move(ee_pos, ascend_pos, self.T_APPROACH)
                self.state = State.ASCEND_PICK
                self._print_state(
                    f"Ascending with fruit (held_grip={self._held_grip:.2f})")

        elif self.state == State.ASCEND_PICK:
            if self._track_trajectory(t, gripper_val=self._held_grip):
                if self.pick_only:
                    # Hold fruit up for 1.5 s so it's visible, then release
                    self._dwell_start = time.time()
                    self._dwell_hold = 1.5
                    self._grip_start_time = time.time() + 1.5  # defer open
                    self._grip_from = float(self._held_grip)
                    self._grip_to = GRIP_OPEN
                    self._grip_duration = self.T_GRIP
                    self.state = State.OPEN_GRIPPER
                    self.sorted_count += 1
                    self._print_state(
                        f"Picked {self.current_target['type']} "
                        f"(#{self.sorted_count}) — holding up"
                    )
                else:
                    basket_above, used_above_z = self._find_reachable_basket_above()
                    if basket_above is None:
                        # No reachable transit pose above this basket. Bail to
                        # DONE with the fruit still held — operator recovers
                        # rather than dropping at the wrong height.
                        print(f"  [warn] basket_above unreachable for "
                              f"{self.current_target['type']} — aborting place")
                        self.state = State.DONE
                        return
                    self._start_move(ee_pos, basket_above, self.T_TRANSIT)
                    self.state = State.MOVE_TO_BASKET
                    self._print_state(
                        f"Moving to {self.current_target['type']} basket "
                        f"(transit z={used_above_z:.3f})")

        elif self.state == State.MOVE_TO_BASKET:
            if self._track_trajectory(t, gripper_val=self._held_grip):
                # Pre-validate the descend target: if IK fails at the basket's
                # nominal z, raise z by 1 cm steps until it solves. Without
                # this guard, a None target_joints lets _track_trajectory
                # short-circuit and the gripper opens at the previous (high)
                # pose — banana basket at large -X reproduced this 2026-04-28.
                place_pos, used_place_z = self._find_reachable_place_pos()
                if place_pos is None:
                    print(f"  [warn] no reachable place height for "
                          f"{self.current_target['type']} basket — aborting")
                    self.state = State.DONE
                    return
                self._start_move(ee_pos, place_pos, self.T_APPROACH)
                self.state = State.DESCEND_PLACE
                self._print_state(f"Descending to basket (z={used_place_z:.3f})")

        elif self.state == State.DESCEND_PLACE:
            if self._track_trajectory(t, gripper_val=self._held_grip):
                self._start_grip_interp(GRIP_OPEN, duration=self.T_GRIP)
                self.state = State.OPEN_GRIPPER
                self._print_state("Opening gripper (ramped)")

        elif self.state == State.OPEN_GRIPPER:
            # Ramp gripper from held value down to GRIP_OPEN, avoids
            # yanking the servo past its open endstop (-1289 overload).
            g = self._update_grip_interp()
            self._execute_position(ee_pos, g)
            if self._grip_interp_done():
                self._held_grip = GRIP_OPEN
                if self.pick_only:
                    # Go straight home, skip basket descent
                    self._traj_end_pos = None
                    self.state = State.GO_HOME
                    self._print_state("Released — returning home")
                else:
                    self.sorted_count += 1
                    ascend_pos = ee_pos.copy()
                    ascend_pos[2] = self.SAFE_Z
                    self._start_move(ee_pos, ascend_pos, self.T_APPROACH)
                    self.state = State.ASCEND_PLACE
                    self._print_state(
                        f"Placed {self.current_target['type']} "
                        f"(#{self.sorted_count})"
                    )

        elif self.state == State.ASCEND_PLACE:
            if self._track_trajectory(t, gripper_val=GRIP_OPEN):
                self._traj_end_pos = None
                if not self.fruit_queue:
                    # Single-pick path (picker_viewer's pick_single): skip the
                    # HOME_POS detour and let the caller drive directly to
                    # survey1 for the next photo. Saves ~3 s per pick.
                    self.state = State.DONE
                    self._print_state("Place complete — returning to caller")
                else:
                    # Multi-fruit autonomous path: still park at HOME between
                    # fruits as a safety pose.
                    self.state = State.GO_HOME
                    self._print_state("Returning home for next fruit")

    def _start_move(self, start, end, duration):
        """Set up a new trajectory.

        2026-04-23 refactor: interpolate in JOINT space (not Cartesian).
        A straight line in Cartesian can pass through singular regions or
        trigger IK solution-branch flips, causing _execute_position to
        fail silently and the arm to stall mid-descent. `slow_move_to_joints`
        uses joint-space linear interp and is field-proven; mirror it here.
        IK is evaluated ONCE at the endpoint.
        """
        self._traj_start = time.time()
        self._traj_duration = duration
        self._traj_start_pos = start.copy() if start is not None else None
        self._traj_end_pos = end.copy()
        # Pre-compute target joints; _ik_safe returns None if unreachable.
        target_joints = self._ik_safe(end)
        self._traj_target_joints = (None if target_joints is None
                                      else np.asarray(target_joints, dtype=float))
        start_joints, _ = self.qarm.read_all()
        self._traj_start_joints = np.asarray(start_joints, dtype=float).copy()

    def _move_complete(self, t):
        """Check if current trajectory is complete."""
        return (time.time() - self._traj_start) >= self._traj_duration

    def _track_trajectory(self, t, gripper_val):
        """Joint-space smoothstep interp followed by a settle phase that
        holds the target for T_SETTLE seconds. The settle matches
        slow_move_to_joints' trailing hammer-loop — without it the arm is
        handed off to CLOSE_GRIPPER while still lagging the final setpoint
        (position-mode PID has ~0.5-1.0 s tracking lag on big descents).

        If target joints weren't solvable, hold last pose and return True
        so the FSM advances rather than spinning forever. Upstream callers
        should have pre-validated reachability via _ik_safe.
        """
        if self._traj_target_joints is None:
            return True
        elapsed = time.time() - self._traj_start
        total = self._traj_duration + self.T_SETTLE
        if elapsed >= total:
            self.qarm.set_joints_and_gripper(
                self._traj_target_joints, gripper_val)
            return True
        if elapsed >= self._traj_duration:
            # Settle phase: keep commanding target so arm physically reaches it.
            self.qarm.set_joints_and_gripper(
                self._traj_target_joints, gripper_val)
            return False
        s = elapsed / self._traj_duration
        # smoothstep: zero velocity at both endpoints.
        s = 3.0 * s * s - 2.0 * s * s * s
        joints = (self._traj_start_joints
                  + s * (self._traj_target_joints - self._traj_start_joints))
        self.qarm.set_joints_and_gripper(joints, gripper_val)
        return False

    def _compute_pick_z(self, fruit_xyz, fruit_type=None):
        """Z to descend to for a pick, clamped to the per-type floor.
        PICK_OFFSET = how far below the fruit centroid we aim so the gripper
        straddles the fruit rather than closing on thin air above it.
        Per-fruit-type overrides live in PICK_DEPTH_OVERRIDES."""
        overrides = self.PICK_DEPTH_OVERRIDES.get(fruit_type, {})
        offset = overrides.get("pick_offset", self.PICK_OFFSET)
        floor = overrides.get("pick_z_floor", self.PICK_Z)
        return max(floor, float(fruit_xyz[2]) - offset)

    def _start_grip_interp(self, target, duration=None):
        """Begin a smooth gripper open/close interpolation from the current
        held value to target. Replaces snap-to-target which stalls the
        servo against objects or endstops."""
        self._grip_from = float(self._held_grip)
        self._grip_to = float(target)
        self._grip_start_time = time.time()
        self._grip_duration = duration if duration is not None else self.T_GRIP

    def _update_grip_interp(self):
        """Advance the gripper interpolation and return the current value
        to command. Called every loop iteration during open/close states."""
        if self._grip_duration <= 0:
            self._held_grip = self._grip_to
            return self._grip_to
        elapsed = time.time() - self._grip_start_time
        frac = min(1.0, max(0.0, elapsed / self._grip_duration))
        s = 3 * frac * frac - 2 * frac * frac * frac  # smoothstep
        g = self._grip_from + s * (self._grip_to - self._grip_from)
        return g

    def _grip_interp_done(self):
        return (time.time() - self._grip_start_time) >= self._grip_duration

    def set_gripper_ramp(self, target, duration=None):
        """Synchronously ramp the gripper from the current held value to
        `target` over `duration` seconds (default T_GRIP). After the ramp,
        reads actual gripper position from the driver and stores it in
        `_held_grip` so subsequent commands do not fight a stalled servo.

        Thread-safety: blocking, intended for teach-pendant / remote use.
        Do NOT call from inside `run_autonomous`'s FSM step loop; the FSM
        has its own stepwise ramp via _update_grip_interp.
        """
        self._start_grip_interp(target, duration=duration)
        # Hold current joints while ramping so only the gripper moves.
        # Send joints directly (bypass IK) since we already have joint angles,
        # not an xyz target.
        joints = self._joints_snapshot()
        while not self._grip_interp_done():
            g = self._update_grip_interp()
            self.qarm.set_joints_and_gripper(joints, g)
            time.sleep(0.01)
        # Settle for a few frames, then read back actual.
        for _ in range(3):
            self.qarm.set_joints_and_gripper(joints, target)
            time.sleep(0.02)
        try:
            _, actual = self.qarm.read_all()
            self._held_grip = float(actual)
        except Exception:
            self._held_grip = float(target)
        return self._held_grip

    def _joints_snapshot(self):
        joints, _ = self.qarm.read_all()
        return np.asarray(joints, dtype=float)

    def _grip_close_for(self, fruit_type):
        """Return the close-gripper target for `fruit_type`, falling back to
        the global GRIP_CLOSE. Result is clamped to [0, 1] so future
        overrides can't drive the servo past its endstop."""
        target = self.GRIP_CLOSE_OVERRIDES.get(fruit_type, GRIP_CLOSE)
        return float(min(max(target, 0.0), 1.0))

    def _find_reachable_basket_above(self):
        """Search for a transit pose above the active basket whose IK solves.

        Tries SAFE_Z first, then steps up by 1 cm to SAFE_Z+0.10. Returns
        (pos_xyz, z) on success, or (None, None) if every step fails.
        Falling back to a higher Z covers baskets near the workspace edge
        where the SAFE_Z plane intersects the IK-singular region.
        """
        if self.target_basket is None:
            return None, None
        Z_MAX = self.SAFE_Z + 0.10
        z = float(self.SAFE_Z)
        while z <= Z_MAX + 1e-6:
            candidate = self.target_basket.copy()
            candidate[2] = z
            if self._ik_safe(candidate) is not None:
                return candidate, z
            z += 0.01
        return None, None

    def _find_reachable_place_pos(self):
        """Search for the lowest reachable release pose at or above the
        basket's taught z, capped at SAFE_Z so the fruit is never dropped
        from above transit height. Returns (pos_xyz, z) or (None, None).

        Fixes the failure mode where a None IK at the taught z let the
        FSM short-circuit DESCEND_PLACE and open the gripper at the prior
        (transit) pose.
        """
        if self.target_basket is None:
            return None, None
        z = float(self.target_basket[2])
        while z <= self.SAFE_Z + 1e-6:
            candidate = self.target_basket.copy()
            candidate[2] = z
            if self._ik_safe(candidate) is not None:
                return candidate, z
            z += 0.01
        return None, None

    def _ik_safe(self, target_pos):
        """Try IK for target_pos. Returns joints array or None on failure
        (IK divergence, unreachable, or joint-limit violation).
        Use this for pre-flight reachability checks before committing to
        a trajectory."""
        try:
            phi = inverse_kinematics(target_pos, gamma=0.0)
        except Exception:
            return None
        phi = np.array(phi, dtype=float)
        limits = getattr(self.qarm, "JOINT_LIMITS", None)
        if limits is not None:
            for i in range(4):
                lo, hi = limits[i]
                if phi[i] < lo or phi[i] > hi:
                    return None
        return phi

    def _execute_position(self, target_pos, gripper_val):
        """Compute IK and send commands to the robot. On IK failure the
        arm holds its last commanded pose (position-mode PID) and a warning
        is logged once. Callers should pre-validate with _ik_safe if they
        need to know a target is reachable before committing."""
        phi = self._ik_safe(target_pos)
        if phi is None:
            if not getattr(self, "_ik_warn_fired", False):
                print(f"  [warn] IK failed for target "
                      f"{np.round(target_pos,3)} — holding last pose")
                self._ik_warn_fired = True
            return False
        self._ik_warn_fired = False
        self.qarm.set_joints_and_gripper(phi, gripper_val)
        return True

    def _print_state(self, msg):
        """Print state transition message."""
        print(f"  [{self.state.name:20s}] {msg}")


# Module-level wrappers for Stateflow entry actions. Stateflow stores
# state IDs; the actual compute lives here so Python tests cover it.

_stateflow_ctx = {'controller': None}


def stateflow_init(fruit_positions=None, fruit_types=None,
                    pick_only=False):
    """Called once by Stateflow's INIT entry action. Builds the shared
    controller instance."""
    from qarm_driver import QArmDriver
    q = QArmDriver(); q.connect()
    c = FruitSortingController(q, pick_only=pick_only)
    if fruit_positions is not None:
        c.set_fruit_positions(fruit_positions, fruit_types)
    _stateflow_ctx['controller'] = c
    return True


def stateflow_select(joints_cur):
    """Pick the next reachable fruit or signal 'queue empty'. Returns
    the approach-phi vector (length 4) or an empty list when queue is
    empty / no reachable fruit."""
    c = _stateflow_ctx.get('controller')
    if c is None or not c.fruit_queue:
        return []
    # Reuse the pre-flight reachability test already in
    # _step -> SELECT_FRUIT. Pull the next fruit, check, skip if bad.
    while c.fruit_queue:
        target = c.fruit_queue[0]
        approach = target['pos'].copy(); approach[2] = c.APPROACH_Z
        pick = target['pos'].copy()
        pick[2] = c._compute_pick_z(target['pos'], fruit_type=target['type'])
        if c._ik_safe(approach) is None or c._ik_safe(pick) is None:
            c.fruit_queue.pop(0)
            if c.logger: c.logger.log("PRE_FLIGHT_SKIP",
                                        type=target['type'])
            continue
        c.current_target = c.fruit_queue.pop(0)
        c.target_basket = c.BASKETS.get(c.current_target['type'],
                                          c.BASKETS['tomato'])
        phi = c._ik_safe(approach)
        return phi.tolist()
    return []


def stateflow_close_grip():
    """Ramp gripper closed, return settled value. Blocking.
    Uses per-fruit-type override when current_target is set."""
    c = _stateflow_ctx.get('controller')
    if c is None: return 0.15
    fruit_type = (c.current_target or {}).get('type')
    return c.set_gripper_ramp(c._grip_close_for(fruit_type))


def stateflow_open_grip():
    c = _stateflow_ctx.get('controller')
    if c is None: return GRIP_OPEN
    return c.set_gripper_ramp(GRIP_OPEN)


def stateflow_sorted_count():
    c = _stateflow_ctx.get('controller')
    return c.sorted_count if c is not None else 0


def fruit_queue():
    """Exposed for Stateflow transition guards."""
    c = _stateflow_ctx.get('controller')
    return c.fruit_queue if c is not None else []
