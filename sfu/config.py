"""Experiment configuration: dataclasses + validation + JSON IO."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

from .calibration import Monitor
from .distributions import CONDITIONS


@dataclass
class StimulusConfig:
    contrast: float = 0.5                 # Michelson, relative to mid-gray background
    envelope: Literal["fixed_deg", "fixed_cycles"] = "fixed_deg"
    envelope_sd_deg: float = 0.75         # Gaussian envelope SD if fixed_deg
    envelope_sd_cycles: float = 1.5       # envelope SD in cycles if fixed_cycles (size then varies with SF)
    eccentricity_deg: float = 5.0         # target eccentricity (0 = fixation)
    polar_angle_deg: float = 0.0          # 0 = right horizontal meridian, 180 = left
    base_orientation_deg: float = 0.0     # reference orientation (0 = vertical in PsychoPy)
    tilt_deg: float = 3.0                 # +/- tilt from reference when no calibration block is used
    randomize_phase: bool = True
    duration_ms: float = 100.0
    fixation_ms: float = 500.0
    response_window_ms: float = 1500.0
    iti_ms: tuple[float, float] = (800.0, 1400.0)  # uniform jitter range
    fixation_size_deg: float = 0.3
    background_rgb: float = 0.0           # PsychoPy rgb units; 0 = mid gray


@dataclass
class DistributionConfig:
    sf_min_cpd: float = 1.0
    sf_max_cpd: float = 4.0
    n_levels: int = 9
    matching: Literal["mode_matched", "moment_matched"] = "mode_matched"
    narrow_sd: float = 0.2                # octaves
    broad_sd: float = 0.5
    skew_alpha: float = 6.0
    bimodal_sep: float = 1.0              # octaves between the two modes (initial guess if moment-matched)
    bimodal_sd: float = 0.2
    target_sd: float | None = None        # moment-matched target SD (octaves); None = broad Gaussian's
    conditions: list[str] = field(default_factory=lambda: list(CONDITIONS))
    custom: dict = field(default_factory=dict)  # {name: {"family":..., "params": {...}}}


@dataclass
class DesignConfig:
    paradigm: Literal["orientation", "probe"] = "orientation"
    sampling: Literal["fixed_count", "probability"] = "fixed_count"
    trials_per_block: int = 108
    blocks_per_condition: int = 1
    counterbalance: Literal["williams", "fixed", "random"] = "williams"
    practice_trials: int = 12
    calibration_block: bool = False       # per-SF staircases under uniform prior, then frozen tilt
    calibration_trials_per_level: int = 30
    staircase_start_tilt: float = 8.0
    staircase_step: float = 1.0           # multiplicative step in log2 units of tilt (1 = factor 2 at start)
    staircase_min_step: float = 0.125
    cue: Literal["text", "histogram", "symbol", "none"] = "histogram"
    probe_rate: float = 0.5               # probe paradigm: fraction of trials with a probe
    probe_sf_min_cpd: float = 0.5         # probes should span beyond the expected range to estimate tuning
    probe_sf_max_cpd: float = 8.0
    probe_n_levels: int = 9
    trigger: Literal["none", "keyboard", "serial"] = "none"
    trigger_key: str = "5"
    markers: Literal["none", "log", "serial"] = "none"
    serial_port: str = ""


@dataclass
class ExperimentConfig:
    study_seed: int = 20251117
    monitor: Monitor = field(default_factory=lambda: Monitor(53.1, 57.0, 1920, 1080, 60.0, calibrated=False, name="demo-uncalibrated"))
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    distributions: DistributionConfig = field(default_factory=DistributionConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    key_left: str = "left"                # response for counter-clockwise tilt
    key_right: str = "right"              # response for clockwise tilt
    key_quit: str = "escape"
    keyboard_backend: Literal["ptb", "iohub", "event"] = "ptb"  # ptb: best timing; needs macOS Input Monitoring permission
    output_dir: str = "data"
    min_px_per_cycle: float = 4.0
    max_frame_rate_error: float = 0.01    # abort if measured refresh deviates by >1%

    def validate(self) -> "ExperimentConfig":
        s, d, g = self.stimulus, self.distributions, self.design
        err = []
        if not 0 < s.contrast <= 1: err.append("stimulus.contrast must be in (0, 1]")
        if s.duration_ms <= 0 or s.response_window_ms <= 0: err.append("durations must be > 0")
        if not (0 <= s.iti_ms[0] <= s.iti_ms[1]): err.append("stimulus.iti_ms must be (lo, hi) with 0 <= lo <= hi")
        if not 0 < d.sf_min_cpd < d.sf_max_cpd: err.append("need 0 < sf_min_cpd < sf_max_cpd")
        if d.n_levels < 2: err.append("n_levels >= 2")
        if not d.conditions: err.append("distributions.conditions must not be empty")
        if g.trials_per_block < 1 or g.blocks_per_condition < 1: err.append("trials_per_block and blocks_per_condition must be >= 1")
        unknown = set(d.conditions) - set(CONDITIONS) - set(d.custom)
        if unknown: err.append(f"unknown conditions {sorted(unknown)}")
        if g.trials_per_block < 2 * d.n_levels and g.sampling == "fixed_count":
            err.append("trials_per_block too small for fixed_count balancing (< 2*n_levels)")
        if not 0 <= g.probe_rate <= 1: err.append("probe_rate in [0, 1]")
        if not 0 < g.probe_sf_min_cpd < g.probe_sf_max_cpd or g.probe_n_levels < 2: err.append("bad probe SF range")
        if g.trigger == "serial" or g.markers == "serial":
            if not g.serial_port: err.append("serial trigger/markers need design.serial_port")
        if self.key_left == self.key_right: err.append("response keys must differ")
        if err:
            raise ValueError("invalid config:\n  " + "\n  ".join(err))
        sf_hi = max(d.sf_max_cpd, g.probe_sf_max_cpd if g.paradigm == "probe" else 0)
        self.monitor.check_sf(sf_hi, self.min_px_per_cycle)
        # largest stimulus must fit on screen
        half_w = self.monitor.width_px / 2
        reach = abs(self.monitor.ecc2px(s.eccentricity_deg)) + self.monitor.deg2px(6 * max_envelope_sd_deg(self)) / 2
        if reach > half_w:
            raise ValueError(f"stimulus extends off-screen ({reach:.0f} px > {half_w:.0f} px from centre)")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def max_envelope_sd_deg(cfg: ExperimentConfig) -> float:
    s = cfg.stimulus
    lo = cfg.distributions.sf_min_cpd
    if cfg.design.paradigm == "probe":
        lo = min(lo, cfg.design.probe_sf_min_cpd)
    return s.envelope_sd_deg if s.envelope == "fixed_deg" else s.envelope_sd_cycles / lo


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    extra = set(d) - names
    if extra:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(extra)}")
    return cls(**d)


def from_dict(d: dict) -> ExperimentConfig:
    d = dict(d)
    sub = {"monitor": Monitor, "stimulus": StimulusConfig, "distributions": DistributionConfig, "design": DesignConfig}
    for k, cls in sub.items():
        if k in d:
            d[k] = _build(cls, d[k])
    if "stimulus" in d and isinstance(d["stimulus"].iti_ms, list):
        d["stimulus"].iti_ms = tuple(d["stimulus"].iti_ms)
    return _build(ExperimentConfig, d).validate()


def load(path: str | Path) -> ExperimentConfig:
    return from_dict(json.loads(Path(path).read_text()))
