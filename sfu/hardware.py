"""Scanner trigger and event-marker hooks (log, serial port, Lab Streaming Layer).
Hardware libraries are imported only when used. LSL markers are tested against an
LSL inlet; serial output and scanner triggers are not yet tested on hardware.
"""
from __future__ import annotations

import logging

log = logging.getLogger("sfu.hardware")

EVENT_CODES = {"block_start": 10, "fixation": 20, "stim_on": 30, "stim_off": 31,
               "response": 50, "block_end": 90}


class Markers:
    """Null marker sink. Subclass and override `send` for real hardware."""

    def send(self, event: str, code: int | None = None, info: str = "") -> None:
        pass

    def close(self) -> None:
        pass


class LogMarkers(Markers):
    def send(self, event, code=None, info=""):
        log.info("marker %s=%s %s", event, EVENT_CODES.get(event, code), info)


class SerialMarkers(Markers):
    def __init__(self, port: str, baud: int = 115200):
        import serial  # pyserial; install only if needed
        self.s = serial.Serial(port, baud, timeout=0)

    def send(self, event, code=None, info=""):
        self.s.write(bytes([EVENT_CODES.get(event, code or 0) & 0xFF]))

    def close(self):
        self.s.close()


class LSLMarkers(Markers):
    """Lab Streaming Layer string marker stream, e.g. "stim_on trial=12 cond=narrow sf=2.000".
    LSL stamps each sample with its own clock when it is pushed; LabRecorder or any
    LSL inlet aligns it with the EEG stream."""

    def __init__(self, name: str = "sfu_markers"):
        import pylsl  # install only if needed
        info = pylsl.StreamInfo(name, "Markers", 1, pylsl.IRREGULAR_RATE, "string", name)
        self.outlet = pylsl.StreamOutlet(info)

    def send(self, event, code=None, info=""):
        self.outlet.push_sample([f"{event} {info}".strip()])

    def close(self):
        del self.outlet


def make_markers(kind: str, port: str = "", lsl_name: str = "sfu_markers") -> Markers:
    if kind == "serial":
        return SerialMarkers(port)
    if kind == "lsl":
        return LSLMarkers(lsl_name)
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
