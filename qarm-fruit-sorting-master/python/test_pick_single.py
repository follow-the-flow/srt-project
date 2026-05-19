"""Tests for FruitSortingController.pick_single synchronous entry point."""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sorting_controller import FruitSortingController, State
from test_integration import MockQArm


def _compressed(c):
    c.T_TRANSIT = 0.05; c.T_APPROACH = 0.05; c.T_PICK = 0.05
    c.T_DWELL = 0.02; c.T_GRIP = 0.02; c.T_SETTLE = 0.02
    return c


def test_pick_single_reachable_returns_true():
    name = "pick_single_reachable"
    q = MockQArm()
    c = _compressed(FruitSortingController(q, pick_only=True))
    ok = c.pick_single(np.array([0.35, 0.10, 0.13]), "strawberry", dt=0.001)
    assert ok, "pick_single returned False for reachable fruit"
    assert c.sorted_count == 1, f"sorted_count = {c.sorted_count}"
    assert c.state == State.DONE, f"state = {c.state.name}"
    return name, True, "reachable fruit sorted once"


def test_pick_single_unreachable_returns_false():
    name = "pick_single_unreachable"
    q = MockQArm()
    c = _compressed(FruitSortingController(q, pick_only=True))
    ok = c.pick_single(np.array([2.0, 0.0, 0.5]), "tomato", dt=0.001)
    assert not ok, "pick_single returned True for out-of-reach fruit"
    assert c.sorted_count == 0, f"sorted_count = {c.sorted_count}"
    return name, True, "unreachable skipped, sorted_count unchanged"


def test_pick_single_twice_accumulates():
    name = "pick_single_twice"
    q = MockQArm()
    c = _compressed(FruitSortingController(q, pick_only=True))
    assert c.pick_single(np.array([0.35, 0.10, 0.13]), "strawberry", dt=0.001)
    assert c.pick_single(np.array([0.30, -0.05, 0.08]), "tomato", dt=0.001)
    assert c.sorted_count == 2, f"sorted_count = {c.sorted_count}"
    return name, True, "two back-to-back picks sorted_count==2"


def test_pick_single_invokes_tick_observer():
    name = "pick_single_invokes_tick_observer"
    q = MockQArm()
    c = _compressed(FruitSortingController(q, pick_only=True))
    counter = {"n": 0}
    c.tick_observer = lambda: counter.__setitem__("n", counter["n"] + 1)
    ok = c.pick_single(np.array([0.35, 0.10, 0.13]), "strawberry", dt=0.001)
    assert ok, "pick_single returned False"
    assert counter["n"] >= 5, f"observer not called enough: n={counter['n']}"
    return name, True, f"observer fired {counter['n']} times"


def test_tick_observer_exception_does_not_crash_fsm():
    name = "tick_observer_exception_isolated"
    q = MockQArm()
    c = _compressed(FruitSortingController(q, pick_only=True))
    def _bad():
        raise RuntimeError("display blew up")
    c.tick_observer = _bad
    ok = c.pick_single(np.array([0.35, 0.10, 0.13]), "strawberry", dt=0.001)
    assert ok, "pick_single should still complete despite observer raising"
    assert c.sorted_count == 1
    return name, True, "observer error swallowed; FSM completed"


TESTS = [test_pick_single_reachable_returns_true,
         test_pick_single_unreachable_returns_false,
         test_pick_single_twice_accumulates,
         test_pick_single_invokes_tick_observer,
         test_tick_observer_exception_does_not_crash_fsm]


def main():
    fails = 0
    for fn in TESTS:
        try:
            name, ok, msg = fn()
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}  {msg}")
            if not ok: fails += 1
        except Exception as ex:
            import traceback
            fails += 1
            print(f"  [FAIL] {fn.__name__}: {ex}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(main())
