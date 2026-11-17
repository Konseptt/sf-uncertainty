"""Session-level state machine. Trial-internal phases (fixation, stimulus,
response, ITI) are frame loops inside the backend; this machine guards the
session flow and records every transition with a timestamp for the log."""
from __future__ import annotations

import time
from enum import Enum


class State(str, Enum):
    INIT = "init"
    INSTRUCTIONS = "instructions"
    BLOCK_INTRO = "block_intro"
    WAIT_TRIGGER = "wait_trigger"
    TRIALS = "trials"
    BLOCK_END = "block_end"
    DONE = "done"
    ABORTED = "aborted"


ALLOWED = {
    State.INIT: {State.INSTRUCTIONS},
    State.INSTRUCTIONS: {State.BLOCK_INTRO, State.DONE},
    State.BLOCK_INTRO: {State.WAIT_TRIGGER, State.TRIALS},
    State.WAIT_TRIGGER: {State.TRIALS},
    State.TRIALS: {State.BLOCK_END},
    State.BLOCK_END: {State.BLOCK_INTRO, State.DONE},
    State.DONE: set(),
    State.ABORTED: set(),
}


class Aborted(Exception):
    """Raised by a backend when the operator presses the quit key."""


class StateMachine:
    def __init__(self):
        self.state = State.INIT
        self.history: list[tuple[float, str, str]] = []

    def to(self, new: State) -> None:
        if new is not State.ABORTED and new not in ALLOWED[self.state]:
            raise RuntimeError(f"illegal transition {self.state.value} -> {new.value}")
        self.history.append((time.time(), self.state.value, new.value))
        self.state = new
