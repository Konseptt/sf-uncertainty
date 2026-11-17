"""Response collection. Real keyboard input lives in runner.PsychoPyBackend
(psychopy.hardware.keyboard; RT clock reset on the stimulus-onset flip).

* SimulatedObserver: SYNTHETIC responses for headless runs and window smoke
  tests. Its parameters are invented for software testing and carry no
  empirical meaning.
"""
from __future__ import annotations

import math

import numpy as np


class SimulatedObserver:
    """2AFC Weibull observer. Threshold depends on SF (synthetic U-shape around 2 cpd)
    and optionally on the expected probability of the shown SF (synthetic
    `expectation_gain`; 0 = no expectation effect)."""

    def __init__(self, rng: np.random.Generator, base_threshold: float = 2.5, slope: float = 2.0,
                 lapse: float = 0.02, expectation_gain: float = 0.0):
        self.rng, self.thr0, self.beta, self.lapse, self.k = rng, base_threshold, slope, lapse, expectation_gain

    def threshold(self, sf_log2: float, p_expected: float = 0.0) -> float:
        return self.thr0 * (1 + 0.4 * (sf_log2 - 1.0) ** 2) * math.exp(-self.k * p_expected)

    def respond(self, trial: dict, tilt_deg: float, keys: tuple[str, str]) -> tuple[str, float]:
        thr = self.threshold(trial["sf_log2"], trial["p_expected"])
        pc = 0.5 + (0.5 - self.lapse / 2) * (1 - math.exp(-((tilt_deg / thr) ** self.beta)))
        correct = self.rng.random() < pc
        key = trial["correct_key"] if correct else (keys[0] if trial["correct_key"] == keys[1] else keys[1])
        rt = 0.30 + self.rng.normal(0.12, 0.03) + self.rng.exponential(0.15)  # ex-Gaussian, seconds
        return key, float(max(0.15, rt))
