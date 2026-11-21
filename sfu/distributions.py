"""Discrete probability distributions over a log2 spatial-frequency grid.

Continuous densities are evaluated at the grid points and renormalized. All
reported statistics come from the discrete PMF used to sample stimuli.

"Right-skewed" means positive skewness, i.e. a longer tail toward high SF. With
mode matching this puts more probability above the peak. With mean matching the
peak sits below the mean. `stats()` reports `p_above_center` for both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import optimize, stats as sps

Family = Literal["gaussian", "uniform", "skewnormal", "bimodal"]
MATCH_TOL = 1e-6


def make_grid(sf_min: float = 1.0, sf_max: float = 4.0, n_levels: int = 9) -> np.ndarray:
    """log2-spaced SF grid (returned in log2 units)."""
    if not (0 < sf_min < sf_max):
        raise ValueError(f"need 0 < sf_min < sf_max, got {sf_min}, {sf_max}")
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    return np.linspace(np.log2(sf_min), np.log2(sf_max), n_levels)


@dataclass(frozen=True)
class DistSpec:
    family: Family
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"family": self.family, "params": {k: float(v) for k, v in self.params.items()}}


def _density(spec: DistSpec, x: np.ndarray) -> np.ndarray:
    p = spec.params
    if spec.family == "gaussian":
        return sps.norm.pdf(x, p["loc"], p["scale"])
    if spec.family == "uniform":
        return np.ones_like(x)
    if spec.family == "skewnormal":
        return sps.skewnorm.pdf(x, p["alpha"], p["loc"], p["scale"])
    if spec.family == "bimodal":
        d, s, w = p["sep"] / 2, p["scale"], p.get("weight", 0.5)
        return w * sps.norm.pdf(x, p["loc"] - d, s) + (1 - w) * sps.norm.pdf(x, p["loc"] + d, s)
    raise ValueError(f"unknown family {spec.family!r}")


def pmf(spec: DistSpec, grid: np.ndarray) -> np.ndarray:
    d = _density(spec, np.asarray(grid, float))
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        raise ValueError(f"degenerate distribution on grid: {spec}")
    return d / d.sum()


def stats(p: np.ndarray, grid: np.ndarray, center: float | None = None) -> dict:
    """Moments of a discrete PMF over log2-SF grid. Entropy in bits."""
    p, g = np.asarray(p, float), np.asarray(grid, float)
    if not np.isclose(p.sum(), 1.0) or np.any(p < 0):
        raise ValueError("p is not a valid PMF")
    center = float(np.mean(g[[0, -1]])) if center is None else center
    mean = float(p @ g)
    var = float(p @ (g - mean) ** 2)
    sd = np.sqrt(var)
    skew = float(p @ ((g - mean) / sd) ** 3) if sd > 0 else 0.0
    kurt = float(p @ ((g - mean) / sd) ** 4 - 3) if sd > 0 else 0.0
    nz = p[p > 0]
    # modes: strict local maxima (plateaus -> all plateau points if global max)
    pad = np.r_[-np.inf, p, -np.inf]
    loc_max = (pad[1:-1] >= pad[:-2]) & (pad[1:-1] >= pad[2:])
    if np.allclose(p, p[0]):
        modes = []  # uniform: no mode
    else:
        modes = [float(g[i]) for i in np.flatnonzero(loc_max) if p[i] > 1e-3]
    return {
        "mean_log2": mean,
        "mean_cpd_geometric": float(2**mean),
        "var_log2": var,
        "sd_octaves": float(sd),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "entropy_bits": float(-(nz @ np.log2(nz))),
        "max_entropy_bits": float(np.log2(len(g))),
        "modes_log2": modes,
        "modes_cpd": [float(2**m) for m in modes],
        "p_above_center": float(p[g > center + 1e-12].sum()),
        "p_below_center": float(p[g < center - 1e-12].sum()),
    }


def _continuous_mode(spec: DistSpec) -> float:
    p = spec.params
    lo, hi = p["loc"] - 4 * p["scale"], p["loc"] + 4 * p["scale"]
    r = optimize.minimize_scalar(lambda x: -_density(spec, np.array([x]))[0], bounds=(lo, hi), method="bounded")
    return float(r.x)


def mode_matched_skew(alpha: float, scale: float, center: float) -> DistSpec:
    """Skew-normal whose continuous mode sits exactly at `center`."""
    probe = DistSpec("skewnormal", {"alpha": alpha, "loc": 0.0, "scale": scale})
    return DistSpec("skewnormal", {"alpha": alpha, "loc": center - _continuous_mode(probe), "scale": scale})


def match_moments(spec: DistSpec, grid: np.ndarray, target_mean: float, target_var: float,
                  free: tuple[str, str]) -> tuple[DistSpec, dict]:
    """Solve two free parameters so the *discrete* PMF has target mean & variance.

    Returns (spec, report). report['matched'] is True only if both residuals < MATCH_TOL.
    Infeasible targets (e.g. variance beyond what the family can reach on a bounded
    grid) return matched=False with the closest solution found.
    """
    base = dict(spec.params)
    x0 = np.array([base[k] for k in free], float)

    def resid(x):
        q = dict(base, **dict(zip(free, x)))
        if any(q[k] <= 0 for k in ("scale", "sep") if k in q):
            return np.array([1e3, 1e3])
        try:
            pp = pmf(DistSpec(spec.family, q), grid)
        except ValueError:
            return np.array([1e3, 1e3])
        s = stats(pp, grid)
        return np.array([s["mean_log2"] - target_mean, s["var_log2"] - target_var])

    r = optimize.least_squares(resid, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000)
    out = DistSpec(spec.family, dict(base, **dict(zip(free, r.x))))
    res = resid(r.x)
    return out, {"target_mean_log2": target_mean, "target_var_log2": target_var,
                 "residual_mean": float(res[0]), "residual_var": float(res[1]),
                 "matched": bool(np.all(np.abs(res) < MATCH_TOL))}


CONDITIONS = ("narrow", "broad", "uniform", "right_skewed", "left_skewed", "bimodal")


def build_conditions(grid: np.ndarray, matching: Literal["mode_matched", "moment_matched"] = "moment_matched",
                     center: float = 1.0, narrow_sd: float = 0.2, broad_sd: float = 0.5,
                     skew_alpha: float = 6.0, bimodal_sep: float = 1.0, bimodal_sd: float = 0.2,
                     target_sd: float | None = None) -> dict[str, dict]:
    """Build the six standard conditions.

    mode_matched: peaks at `center` (bimodal: peaks symmetric about center); spreads set by
        the given sd / alpha / sep. Means and variances will generally differ.
    moment_matched: narrow = Gaussian(center, narrow_sd) (the uncertainty manipulation);
        broad, skews and bimodal solved to share mean=center and var=target_sd**2
        (default target = discrete variance of the broad Gaussian before matching).
        Uniform is reported as matched only if it numerically is.
    Returns {name: {"spec", "pmf", "stats", "match"}}.
    """
    out: dict[str, dict] = {}

    def add(name, spec, match=None):
        p = pmf(spec, grid)
        out[name] = {"spec": spec, "pmf": p, "stats": stats(p, grid, center), "match": match}

    if matching == "mode_matched":
        add("narrow", DistSpec("gaussian", {"loc": center, "scale": narrow_sd}))
        add("broad", DistSpec("gaussian", {"loc": center, "scale": broad_sd}))
        add("uniform", DistSpec("uniform"))
        add("right_skewed", mode_matched_skew(+skew_alpha, broad_sd, center))
        add("left_skewed", mode_matched_skew(-skew_alpha, broad_sd, center))
        add("bimodal", DistSpec("bimodal", {"loc": center, "sep": bimodal_sep, "scale": bimodal_sd}))
        return out
    if matching != "moment_matched":
        raise ValueError(f"unknown matching {matching!r}")

    if target_sd is None:
        target_var = stats(pmf(DistSpec("gaussian", {"loc": center, "scale": broad_sd}), grid), grid)["var_log2"]
    else:
        target_var = target_sd**2
    add("narrow", DistSpec("gaussian", {"loc": center, "scale": narrow_sd}),
        {"note": "uncertainty manipulation: mean matched, variance intentionally smaller"})
    s, m = match_moments(DistSpec("gaussian", {"loc": center, "scale": broad_sd}), grid, center, target_var, ("loc", "scale"))
    add("broad", s, m)
    u = pmf(DistSpec("uniform"), grid)
    us = stats(u, grid)
    add("uniform", DistSpec("uniform"), {"target_mean_log2": center, "target_var_log2": target_var,
                                          "residual_mean": us["mean_log2"] - center,
                                          "residual_var": us["var_log2"] - target_var,
                                          "matched": bool(abs(us["mean_log2"] - center) < MATCH_TOL and abs(us["var_log2"] - target_var) < MATCH_TOL),
                                          "note": "no free parameters on a fixed grid"})
    for name, a in (("right_skewed", +skew_alpha), ("left_skewed", -skew_alpha)):
        s, m = match_moments(DistSpec("skewnormal", {"alpha": a, "loc": center, "scale": broad_sd}), grid, center, target_var, ("loc", "scale"))
        add(name, s, m)
    s, m = match_moments(DistSpec("bimodal", {"loc": center, "sep": bimodal_sep, "scale": bimodal_sd}), grid, center, target_var, ("loc", "sep"))
    add(name := "bimodal", s, m)
    return out


def stats_table(conds: dict[str, dict]) -> str:
    cols = ("mean_cpd_geometric", "sd_octaves", "entropy_bits", "skewness", "excess_kurtosis", "p_above_center")
    lines = [f"{'condition':<13}" + "".join(f"{c[:12]:>13}" for c in cols) + f"{'modes_cpd':>18}{'matched':>9}"]
    for n, c in conds.items():
        s = c["stats"]
        m = c["match"].get("matched", "-") if c["match"] else "-"
        modes = ",".join(f"{v:.2f}" for v in s["modes_cpd"]) or "none"
        lines.append(f"{n:<13}" + "".join(f"{s[k]:>13.4f}" for k in cols) + f"{modes:>18}{str(m):>9}")
    return "\n".join(lines)


if __name__ == "__main__":
    g = make_grid()
    for mode in ("mode_matched", "moment_matched"):
        print(f"\n== {mode} ==")
        print(stats_table(build_conditions(g, mode)))
