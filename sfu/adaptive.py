"""Per-SF 2-down/1-up orientation staircases (about 71% correct).

Run only in the calibration block with a uniform prior. The resulting tilts are
then fixed for the main blocks, because a staircase running there would absorb
part of the expectation effect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Staircase:
    tilt: float = 8.0          # deg, current
    step: float = 1.0          # log2 units
    min_step: float = 0.125
    min_tilt: float = 0.1
    max_tilt: float = 45.0
    n_correct: int = 0
    last_dir: int = 0
    reversals: list[float] = field(default_factory=list)
    history: list[tuple[float, bool]] = field(default_factory=list)

    def update(self, correct: bool) -> None:
        self.history.append((self.tilt, bool(correct)))
        if correct:
            self.n_correct += 1
            if self.n_correct < 2:
                return
            self.n_correct, direction = 0, -1
        else:
            self.n_correct, direction = 0, +1
        if self.last_dir and direction != self.last_dir:
            self.reversals.append(self.tilt)
            self.step = max(self.min_step, self.step / 2)
        self.last_dir = direction
        self.tilt = min(self.max_tilt, max(self.min_tilt, self.tilt * 2 ** (direction * self.step)))

    def threshold(self, n_last: int = 6) -> float:
        """Geometric mean of the last reversals (falls back to current tilt)."""
        r = self.reversals[-n_last:] or [self.tilt]
        return math.exp(sum(map(math.log, r)) / len(r))


if __name__ == "__main__":
    import random
    rnd = random.Random(1)
    s = Staircase()
    for _ in range(200):  # simulated observer, 2AFC Weibull, threshold 2 deg
        p = 0.5 + 0.5 * (1 - math.exp(-((s.tilt / 2.0) ** 2)))
        s.update(rnd.random() < p)
    assert 1.0 < s.threshold() < 4.0, s.threshold()
    print(f"threshold ~ {s.threshold():.2f} deg, {len(s.reversals)} reversals")
