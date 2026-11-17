"""Monitor geometry and visual-angle conversion.

Exact geometry (no small-angle approximation): a stimulus of `deg` degrees
centred on the line of sight spans 2*d*tan(deg/2) cm. For eccentric stimuli,
the position uses d*tan(ecc). Pixels-per-degree reported at screen centre.

`calibrated` must be set True only after the width, resolution and viewing
distance have been physically measured (and, ideally, luminance gamma-corrected).
Nothing in this module can verify that; it only records the claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Monitor:
    width_cm: float
    distance_cm: float
    width_px: int
    height_px: int
    refresh_hz: float
    calibrated: bool = False
    gamma: float | None = None  # None -> assume hardware/OS linearized; record what was done
    name: str = "unnamed"

    def __post_init__(self):
        for k in ("width_cm", "distance_cm", "width_px", "height_px", "refresh_hz"):
            if getattr(self, k) <= 0:
                raise ValueError(f"monitor.{k} must be > 0")

    @property
    def cm_per_px(self) -> float:
        return self.width_cm / self.width_px

    @property
    def px_per_deg(self) -> float:
        """Pixels per degree at screen centre (exact for a 1 deg centred extent)."""
        return 2 * self.distance_cm * math.tan(math.radians(0.5)) / self.cm_per_px

    def deg2px(self, deg: float) -> float:
        """Extent of a centred stimulus of `deg` degrees, in px."""
        return 2 * self.distance_cm * math.tan(math.radians(deg / 2)) / self.cm_per_px

    def ecc2px(self, ecc_deg: float) -> float:
        """Signed screen offset (px) of a point at eccentricity `ecc_deg` along one axis."""
        return self.distance_cm * math.tan(math.radians(ecc_deg)) / self.cm_per_px

    def cpd2cpp(self, cpd: float) -> float:
        """cycles/degree -> cycles/pixel (centre-of-screen ppd)."""
        return cpd / self.px_per_deg

    def frames(self, ms: float) -> int:
        """Duration in ms -> integer frame count (>=1)."""
        return max(1, round(ms * self.refresh_hz / 1000))

    def check_sf(self, sf_max_cpd: float, min_px_per_cycle: float = 4.0) -> None:
        """Raise if the highest SF is under-sampled by the display."""
        ppc = self.px_per_deg / sf_max_cpd
        if ppc < min_px_per_cycle:
            raise ValueError(f"{sf_max_cpd} cpd = {ppc:.2f} px/cycle at {self.px_per_deg:.1f} px/deg "
                             f"(< {min_px_per_cycle}); increase viewing distance or resolution")


if __name__ == "__main__":
    m = Monitor(width_cm=53.1, distance_cm=57.0, width_px=1920, height_px=1080, refresh_hz=60)
    assert abs(m.deg2px(1) - m.px_per_deg) < 1e-9
    assert m.frames(100) == 6
    print(f"{m.px_per_deg:.2f} px/deg; 4 cpd -> {m.px_per_deg/4:.1f} px/cycle")
