"""Tests for picker_viewer pure helpers."""
import os, sys
import time
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fruit_detector import Detection
from picker_viewer import (
    _filter_by_type, _annotate, _hud_text,
)


def _mk_det(ftype, cx, cy, conf=0.8):
    return Detection(
        fruit_type=ftype,
        center_px=(cx, cy),
        center_base_m=np.array([0.3, 0.1, 0.02]),
        confidence=conf,
        area_px=500,
        bbox=(cx - 10, cy - 10, 20, 20),
    )


def test_filter_by_type():
    name = "filter_by_type"
    dets = [_mk_det("tomato", 1, 1), _mk_det("banana", 2, 2),
            _mk_det("tomato", 3, 3)]
    got = _filter_by_type(dets, "tomato")
    assert len(got) == 2 and all(d.fruit_type == "tomato" for d in got)
    return name, True, "keeps only matching type"


def test_annotate_returns_image_same_shape():
    name = "annotate_shape"
    img = np.full((720, 1280, 3), 50, dtype=np.uint8)
    dets = [_mk_det("tomato", 100, 100), _mk_det("banana", 300, 200)]
    out = _annotate(img, dets, residual_mm=1.2, warnings=["test warn"])
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Annotation must modify pixels (bbox draw)
    assert not np.array_equal(out, img), "annotate left image unchanged"
    return name, True, "shape preserved, pixels modified"


def test_hud_text_contains_counts():
    name = "hud_text"
    msg = _hud_text(n_fruits=5, residual_mm=1.8, mode="idle")
    assert "5" in msg and "1.8" in msg
    return name, True, f"'{msg}'"


class FakeController:
    def __init__(self, succeed=True):
        self.calls = []
        self._succeed = succeed
        self.sorted_count = 0
    def pick_single(self, base_xyz, fruit_type, dt=0.01):
        self.calls.append((np.asarray(base_xyz).copy(), fruit_type))
        if self._succeed:
            self.sorted_count += 1
            return True
        return False


def test_pick_one_calls_controller():
    from picker_viewer import _pick_one
    name = "pick_one_dispatch"
    det = _mk_det("tomato", 100, 100)
    det.center_base_m = np.array([0.42, 0.11, 0.04])
    ctrl = FakeController(succeed=True)
    ok = _pick_one(ctrl, det)
    assert ok is True
    assert len(ctrl.calls) == 1
    pos, ftype = ctrl.calls[0]
    # tomato gets +2 cm x bias (picker_viewer._TOMATO_X_BIAS_M); the
    # global +5 cm bias is applied later inside controller.pick_single,
    # so the tests only see the per-fruit shift here.
    assert np.allclose(pos, [0.44, 0.11, 0.04])
    assert ftype == "tomato"
    return name, True, "ctrl.pick_single called with det base_m + type"


class FakePickerBatchCtx:
    """Minimal fake driver/camera/session_cal that make capture_fruits
    return a fixed detection list every call, for _pick_category testing."""
    def __init__(self, always_same_det):
        self.det = always_same_det


def test_pick_category_skips_stuck_target():
    """A fruit that pick_single always fails on must be skipped after
    RETRY_LIMIT_PER_TARGET tries, not loop forever."""
    import picker_viewer as pv
    name = "pick_category_skips_stuck"
    det = _mk_det("tomato", 200, 200)
    det.center_base_m = np.array([0.5, 0.1, 0.05])

    class Cam: pass
    class Drv: pass
    class Cal: pass
    class Ctrl:
        HOME_POS = np.array([0.4, 0.0, 0.3])
        sorted_count = 0
        def pick_single(self, xyz, t, dt=0.01):
            return False    # always fails → target becomes stuck

    # Monkey-patch capture_fruits to return the stuck det every time.
    orig = None
    import survey_capture as sc
    orig = sc.capture_fruits
    sc.capture_fruits = lambda d, c, s: ([det], {"warnings": []})
    try:
        n = pv._pick_category(Drv(), Cam(), Cal(), Ctrl(),
                               "tomato", lambda: False)
    finally:
        sc.capture_fruits = orig
    # Must terminate (no hang), sorted nothing.
    assert n == 0, f"picks_done={n}"
    return name, True, "stuck target terminated without hang"


def test_live_feed_populates_latest_then_stops():
    name = "live_feed_populates_latest_then_stops"

    class _StubCam:
        def __init__(self):
            self.calls = 0
        def read(self):
            self.calls += 1
            return (np.full((720, 1280, 3), self.calls % 256, dtype=np.uint8),
                    np.zeros((720, 1280), dtype=np.uint16))

    from picker_viewer import _LiveFeed
    cam = _StubCam()
    feed = _LiveFeed(cam)
    feed.start()
    deadline = time.time() + 0.5
    frame = None
    while time.time() < deadline:
        frame = feed.latest()
        if frame is not None:
            break
        time.sleep(0.01)
    feed.stop()
    assert frame is not None, "latest() returned None within 500 ms"
    assert frame.shape == (720, 1280, 3), f"shape = {frame.shape}"
    assert feed._thread is None, "stop() did not clear thread"
    return name, True, f"latest shape={frame.shape}, cam.calls={cam.calls}"


def test_render_observer_throttles_and_overlays_state():
    name = "render_observer_throttles_and_overlays_state"

    captured = {"imshow": [], "waitKey": 0}
    orig_imshow = cv2.imshow
    orig_waitKey = cv2.waitKey
    cv2.imshow = lambda w, f: captured["imshow"].append((w, f.shape))
    cv2.waitKey = lambda d: (captured.__setitem__("waitKey",
                                                  captured["waitKey"] + 1)
                             or 0)
    try:
        from picker_viewer import _make_render_observer

        class _Feed:
            def latest(self):
                return np.zeros((720, 1280, 3), dtype=np.uint8)

        class _Ctrl:
            class state:
                name = "DESCEND"
            current_target = {"type": "tomato",
                              "pos": np.array([0.4, 0.1, 0.05])}

        observer = _make_render_observer(
            _Feed(), "win", _Ctrl(), fps_limit=30.0)
        observer()                  # first call: renders
        observer()                  # immediate: throttled
        time.sleep(0.05)            # > 1/30 s
        observer()                  # third call: renders again
        assert len(captured["imshow"]) == 2, \
            f"expected 2 renders, got {len(captured['imshow'])}"
        assert captured["imshow"][0][0] == "win"
        assert captured["imshow"][0][1] == (720, 1280, 3)
        assert captured["waitKey"] == 2
    finally:
        cv2.imshow = orig_imshow
        cv2.waitKey = orig_waitKey
    return name, True, f"{len(captured['imshow'])} renders, "\
                        f"{captured['waitKey']} waitKeys"


def test_render_observer_handles_none_target():
    name = "render_observer_handles_none_target"
    orig_imshow = cv2.imshow
    orig_waitKey = cv2.waitKey
    cv2.imshow = lambda w, f: None
    cv2.waitKey = lambda d: 0
    try:
        from picker_viewer import _make_render_observer

        class _Feed:
            def latest(self):
                return np.zeros((720, 1280, 3), dtype=np.uint8)

        class _Ctrl:
            class state:
                name = "GO_HOME"
            current_target = None

        observer = _make_render_observer(_Feed(), "win", _Ctrl())
        observer()  # must not raise
    finally:
        cv2.imshow = orig_imshow
        cv2.waitKey = orig_waitKey
    return name, True, "no exception with current_target=None"


def test_pick_one_starts_and_clears_observer_when_camera_provided():
    name = "pick_one_lifecycle_with_camera"

    class _StubCam:
        def read(self):
            return (np.zeros((720, 1280, 3), dtype=np.uint8),
                    np.zeros((720, 1280), dtype=np.uint16))

    class _StubCtrl:
        def __init__(self):
            self.tick_observer = None
            self.observer_during_pick = "UNSET"
            self.calls = []
            class _State: name = "GO_HOME"
            self.state = _State()
            self.current_target = None

        def pick_single(self, xyz, ftype):
            self.observer_during_pick = self.tick_observer
            self.calls.append((tuple(xyz), ftype))
            return True

    from picker_viewer import _pick_one
    from fruit_detector import Detection

    det = Detection(fruit_type="tomato", center_px=(640, 360),
                     center_base_m=np.array([0.4, 0.0, 0.05]),
                     confidence=0.8, area_px=1000, bbox=(620, 340, 40, 40))
    ctrl = _StubCtrl()
    cam = _StubCam()
    ok = _pick_one(ctrl, det, camera=cam, window="win")
    assert ok, "stub pick_single returned True so _pick_one should too"
    assert ctrl.observer_during_pick is not None, \
        "tick_observer was not set during pick_single"
    assert ctrl.tick_observer is None, \
        "tick_observer not restored after pick_single returned"
    # tomato gets +2 cm x bias (picker_viewer._TOMATO_X_BIAS_M); the
    # global +5 cm bias is applied later inside controller.pick_single,
    # so the stub only sees the per-fruit shift here.
    assert len(ctrl.calls) == 1
    pos, ftype = ctrl.calls[0]
    assert np.allclose(pos, (0.42, 0.0, 0.05)) and ftype == "tomato", \
        f"expected ([0.42, 0, 0.05], 'tomato'), got ({pos}, {ftype!r})"
    return name, True, "observer set during pick, cleared after"


TESTS = [
    test_filter_by_type,
    test_annotate_returns_image_same_shape, test_hud_text_contains_counts,
    test_pick_one_calls_controller,
    test_pick_category_skips_stuck_target,
    test_live_feed_populates_latest_then_stops,
    test_render_observer_throttles_and_overlays_state,
    test_render_observer_handles_none_target,
    test_pick_one_starts_and_clears_observer_when_camera_provided,
]


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
