"""Real PsychoPy window + in-process macOS key events (no Accessibility permission needed).
Regression test for the instructions screen never receiving keys. Skipped without
PsychoPy/pyobjc (e.g. the core .venv); run with .venv-psychopy."""
import sys
import threading
import time

import pytest

if sys.platform != "darwin":
    pytest.skip("macOS-only key injection", allow_module_level=True)
pytest.importorskip("psychopy")
AppKit = pytest.importorskip("AppKit")

from sfu import config as cfgmod  # noqa: E402
from sfu.config import ExperimentConfig  # noqa: E402

KEYCODES = {"a": (0, "a"), "right": (124, ""), "left": (123, "")}


def post_key(nswindow, key: str, delay: float):
    def go():
        time.sleep(delay)
        code, ch = KEYCODES[key]
        app = AppKit.NSApplication.sharedApplication()
        for t in (AppKit.NSEventTypeKeyDown, AppKit.NSEventTypeKeyUp):
            e = AppKit.NSEvent.keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode_(
                t, (0, 0), 0, time.monotonic(), nswindow.windowNumber(), None, ch, ch, False, code)
            app.postEvent_atStart_(e, False)
    threading.Thread(target=go, daemon=True).start()


@pytest.mark.parametrize("kb_backend", ["event"])
def test_instructions_and_trial_receive_keys(kb_backend):
    from sfu.runner import PsychoPyBackend
    from sfu.schedule import build_session
    d = ExperimentConfig().to_dict()
    d["keyboard_backend"] = kb_backend
    d["stimulus"]["response_window_ms"] = 3000
    cfg = cfgmod.from_dict(d)
    be = PsychoPyBackend(cfg, windowed=True, demo=True)
    try:
        nsw = be.win.winHandle._nswindow
        post_key(nsw, "a", 0.5)
        t0 = time.time()
        be.instructions("press a key")  # used to hang forever
        assert time.time() - t0 < 3
        trial = build_session(cfg, "K01", 1, 0)["blocks"][1]["trials"][0]
        post_key(nsw, "right", 1.0)  # after fixation (0.5 s) -> during response window
        res = be.run_trial(trial, tilt=3.0)
        assert res["response"] == "right"
        assert 0 < res["rt_s"] < 2.5
    finally:
        be.close()
