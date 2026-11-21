"""Scanner trigger and event-marker hooks. Hardware libraries are imported only
when used. Not yet tested with a scanner or EEG system; measure latency before
use.
"""
from __future__ import annotations

import logging

log = logging.getLogger("sfu.hardware")

EVENT_CODES = {"block_start": 10, "fixation": 20, "stim_on": 30, "stim_off": 31, "probe_on": 40,
               "response": 50, "block_end": 90}


class Markers:
    """Null marker sink. Subclass and override `send` for real hardware."""

    def send(self, event: str, code: int | None = None) -> None:
        pass

    def close(self) -> None:
        pass


class LogMarkers(Markers):
    def send(self, event, code=None):
        log.info("marker %s=%s", event, EVENT_CODES.get(event, code))


class SerialMarkers(Markers):
    def __init__(self, port: str, baud: int = 115200):
        import serial  # pyserial; install only if needed
        self.s = serial.Serial(port, baud, timeout=0)

    def send(self, event, code=None):
        self.s.write(bytes([EVENT_CODES.get(event, code or 0) & 0xFF]))

    def close(self):
        self.s.close()


def make_markers(kind: str, port: str = "") -> Markers:
    if kind == "serial":
        return SerialMarkers(port)
    return LogMarkers() if kind == "log" else Markers()


def wait_keyboard_trigger(win, key: str, quit_key: str) -> float:
    """Block until scanner trigger arrives as a keypress (most MRI trigger boxes emulate '5').
    Returns trigger time (psychopy clock)."""
    from psychopy import event, core
    from .state import Aborted
    keys = event.waitKeys(keyList=[key, quit_key], timeStamped=core.monotonicClock)
    if keys[0][0] == quit_key:
        raise Aborted()
    return float(keys[0][1])
