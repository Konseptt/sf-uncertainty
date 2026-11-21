"""Gabor parameters for PsychoPy GratingStim and a numpy preview.

In PsychoPy, mask='gauss' with maskParams={'sd': 3} gives an envelope SD of
size/6, so size = 6 * envelope SD. Stimuli use units='pix' with SF converted by
sfu.calibration. Orientation is in degrees clockwise from vertical; phase is in
cycles.
"""
from __future__ import annotations

import math

import numpy as np

from .calibration import Monitor
from .config import ExperimentConfig

MASK_SD = 3  # PsychoPy gauss mask: SD = size / (2 * MASK_SD)


def envelope_sd_deg(cfg: ExperimentConfig, sf_cpd: float) -> float:
    s = cfg.stimulus
    return s.envelope_sd_deg if s.envelope == "fixed_deg" else s.envelope_sd_cycles / sf_cpd


def position_deg(cfg: ExperimentConfig, mirror: bool = False) -> tuple[float, float]:
    """Target (or mirrored probe) position in degrees of eccentricity along x/y."""
    a = math.radians(cfg.stimulus.polar_angle_deg + (180 if mirror else 0))
    e = cfg.stimulus.eccentricity_deg
    return (e * math.cos(a), e * math.sin(a))


def position_px(m: Monitor, pos_deg: tuple[float, float]) -> tuple[float, float]:
    """Screen position of a point at eccentricity hypot(x, y) along polar angle atan2(y, x).
    Radial distance is d*tan(ecc), so the result is exact at any polar angle."""
    ecc = math.hypot(*pos_deg)
    if ecc == 0:
        return (0.0, 0.0)
    k = m.ecc2px(ecc) / ecc
    return (pos_deg[0] * k, pos_deg[1] * k)


def gabor_kwargs(cfg: ExperimentConfig, sf_cpd: float, ori: float, phase: float,
                 contrast: float, pos_deg: tuple[float, float]) -> dict:
    """Arguments for psychopy.visual.GratingStim (units='pix')."""
    m = cfg.monitor
    size_px = m.deg2px(6 * envelope_sd_deg(cfg, sf_cpd))
    return dict(tex="sin", mask="gauss", maskParams={"sd": MASK_SD}, units="pix",
                sf=m.cpd2cpp(sf_cpd), size=size_px, ori=ori, phase=phase,
                contrast=contrast, pos=position_px(m, pos_deg), interpolate=True)


def bandpass_noise(n_px: int, px_per_deg: float, center_cpd: float, bandwidth_oct: float,
                   seed: int) -> np.ndarray:
    """Isotropic band-pass noise, n_px x n_px, values in [-1, 1].

    White Gaussian noise is filtered in the Fourier domain with a log2-Gaussian
    pass band centred on `center_cpd` (FWHM `bandwidth_oct` octaves), then scaled
    so its largest absolute value is 1. Drawn at 1 image pixel per screen pixel.
    """
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n_px, d=1 / px_per_deg)  # cycles/deg
    r = np.hypot(*np.meshgrid(f, f))
    sigma = bandwidth_oct / (2 * math.sqrt(2 * math.log(2)))
    with np.errstate(divide="ignore"):
        h = np.exp(-(np.log2(r / center_cpd)) ** 2 / (2 * sigma**2))
    h[r == 0] = 0.0
    img = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n_px, n_px))) * h))
    return img / np.abs(img).max()


def noise_kwargs(cfg: ExperimentConfig, sf_cpd: float, seed: int, pos_deg: tuple[float, float]) -> dict:
    """Arguments for psychopy.visual.ImageStim showing a band-pass noise probe."""
    m, d = cfg.monitor, cfg.design
    n = max(8, int(round(m.deg2px(6 * envelope_sd_deg(cfg, sf_cpd)))))
    return dict(image=bandpass_noise(n, m.px_per_deg, sf_cpd, d.probe_noise_bandwidth_oct, seed),
                mask="gauss", maskParams={"sd": MASK_SD}, units="pix", size=(n, n),
                contrast=d.probe_contrast, pos=position_px(m, pos_deg), interpolate=False)


def make_gabor(win, cfg: ExperimentConfig, **kw):
    from psychopy import visual  # lazy: headless mode never imports PsychoPy
    return visual.GratingStim(win, **gabor_kwargs(cfg, **kw))


def render_gabor(cfg: ExperimentConfig, sf_cpd: float, ori: float = 0.0, phase: float = 0.0,
                 contrast: float | None = None, field_deg: float | None = None) -> tuple[np.ndarray, float]:
    """Numpy luminance image (-1..1 around mean gray) at the monitor's true px/deg.

    Returns (image, extent_deg). Used for previews only; matches PsychoPy's
    orientation (clockwise from vertical) and phase-in-cycles conventions.
    """
    m = cfg.monitor
    c = cfg.stimulus.contrast if contrast is None else contrast
    sd = envelope_sd_deg(cfg, sf_cpd)
    field_deg = 6 * sd if field_deg is None else field_deg
    n = max(8, int(round(m.deg2px(field_deg))))
    ax = (np.arange(n) - (n - 1) / 2) / m.px_per_deg  # deg
    x, y = np.meshgrid(ax, -ax)
    th = math.radians(ori)
    xr = x * math.cos(th) - y * math.sin(th)  # clockwise rotation of a vertical grating
    img = c * np.sin(2 * np.pi * (sf_cpd * xr + phase)) * np.exp(-(x**2 + y**2) / (2 * sd**2))
    return img, field_deg
