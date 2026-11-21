"""Simulated population SF tuning: tuning scenarios, Gaussian and flexible fits
(split Gaussian, bimodal, B-spline), and parameter and model recovery.

This checks whether the design and models can tell tuning hypotheses apart at a
given noise level. It uses static noisy tuning curves sampled at probe SFs.
Voxel-wise pSFT fitting with HRF convolution belongs in the pSF-Toolbox.

The x axis is log2(SF in cpd), so a Gaussian here is a log-Gaussian in cpd.

    python -m sfu.simulation --out figures/simulation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize


def gauss(x, mu, sigma, amp, base):
    return base + amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def split_gauss(x, mu, sl, sr, amp, base):
    s = np.where(x < mu, sl, sr)
    return base + amp * np.exp(-((x - mu) ** 2) / (2 * s**2))


def bimodal(x, mu1, mu2, sigma, amp, base):
    return base + amp * (np.exp(-((x - mu1) ** 2) / (2 * sigma**2)) + np.exp(-((x - mu2) ** 2) / (2 * sigma**2)))


MODELS = {  # fn, initial guesses, lower bounds, upper bounds
    "gauss": (gauss, [(1, 1, 1, 0.1), (0, 0.6, 1, 0), (2, 1.5, 1, 0)], (-3, 0.1, 0, -1), (5, 4, 5, 2)),
    "split_gauss": (split_gauss, [(1, 1, 1, 1, 0.1), (0.5, 0.6, 1.4, 1, 0), (1.5, 1.4, 0.6, 1, 0)], (-3, 0.1, 0.1, 0, -1), (5, 4, 4, 5, 2)),
    "bimodal": (bimodal, [(0, 2, 0.5, 1, 0.1), (0.5, 1.5, 0.4, 1, 0), (-0.5, 2.5, 0.6, 1, 0)], (-3, -3, 0.1, 0, -1), (5, 5, 2, 5, 2)),
}

SCENARIOS = {
    "baseline":   ("gauss", dict(mu=1.0, sigma=0.9, amp=1.0, base=0.1)),
    "narrow":     ("gauss", dict(mu=1.0, sigma=0.55, amp=1.0, base=0.1)),
    "broad":      ("gauss", dict(mu=1.0, sigma=1.35, amp=1.0, base=0.1)),
    "gain":       ("gauss", dict(mu=1.0, sigma=0.9, amp=1.4, base=0.1)),
    "peak_shift": ("gauss", dict(mu=1.5, sigma=0.9, amp=1.0, base=0.1)),
    "asymmetric": ("split_gauss", dict(mu=1.0, sl=0.55, sr=1.4, amp=1.0, base=0.1)),
    "bimodal":    ("bimodal", dict(mu1=0.4, mu2=1.6, sigma=0.4, amp=0.8, base=0.1)),
}


def probe_x(sf_min=0.25, sf_max=12.0, n=13) -> np.ndarray:
    return np.linspace(np.log2(sf_min), np.log2(sf_max), n)


def response(model: str, params: dict, x: np.ndarray) -> np.ndarray:
    return MODELS[model][0](x, **params)


def _ic(rss: float, n: int, k: int) -> tuple[float, float]:
    ll = -n / 2 * (np.log(2 * np.pi * max(rss, 1e-300) / n) + 1)
    return 2 * k - 2 * ll, k * np.log(n) - 2 * ll


def fit(model: str, x: np.ndarray, y: np.ndarray, n_knots: int = 3) -> dict:
    """Least-squares fit. 'spline' = cubic B-spline with n_knots interior knots."""
    if model == "spline":
        t_int = np.linspace(x.min(), x.max(), n_knots + 2)[1:-1]
        t = np.r_[[x.min()] * 4, t_int, [x.max()] * 4]
        spl = interpolate.make_lsq_spline(x, y, t, k=3)
        rss = float(((spl(x) - y) ** 2).sum())
        k = len(spl.c)
        aic, bic = _ic(rss, len(x), k)
        return {"model": model, "params": {"coef": spl.c.tolist()}, "rss": rss, "k": k, "aic": aic, "bic": bic, "predict": spl}
    fn, starts, lo, hi = MODELS[model]
    best = None
    for p0 in starts:
        try:
            p, _ = optimize.curve_fit(fn, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rss = float(((fn(x, *p) - y) ** 2).sum())
        if best is None or rss < best[1]:
            best = (p, rss)
    if best is None:
        raise RuntimeError(f"{model} fit failed")
    names = fn.__code__.co_varnames[1:fn.__code__.co_argcount]
    aic, bic = _ic(best[1], len(x), len(best[0]))
    return {"model": model, "params": dict(zip(names, map(float, best[0]))), "rss": best[1], "k": len(best[0]),
            "aic": aic, "bic": bic, "predict": lambda xx, p=best[0]: fn(xx, *p)}


def parameter_recovery(n_sims: int = 200, noise_sd: float = 0.1, x: np.ndarray | None = None, seed: int = 0) -> dict:
    """Gaussian generator -> Gaussian fit. Returns true/recovered arrays + r, bias, RMSE per parameter."""
    x = probe_x() if x is None else x
    rng = np.random.default_rng(seed)
    true, rec = [], []
    for _ in range(n_sims):
        p = dict(mu=rng.uniform(0.2, 1.8), sigma=rng.uniform(0.5, 1.5), amp=rng.uniform(0.5, 1.5), base=rng.uniform(0, 0.3))
        y = gauss(x, **p) + rng.normal(0, noise_sd, len(x))
        f = fit("gauss", x, y)
        true.append(list(p.values())); rec.append([f["params"][k] for k in p])
    true, rec = np.array(true), np.array(rec)
    names = ["mu", "sigma", "amp", "base"]
    summ = {n: dict(r=float(np.corrcoef(true[:, i], rec[:, i])[0, 1]), bias=float((rec[:, i] - true[:, i]).mean()),
                    rmse=float(np.sqrt(((rec[:, i] - true[:, i]) ** 2).mean()))) for i, n in enumerate(names)}
    return {"names": names, "true": true, "recovered": rec, "summary": summ, "noise_sd": noise_sd}


def model_recovery(n_sims: int = 40, noise_sd: float = 0.1, x: np.ndarray | None = None, seed: int = 1,
                   candidates=("gauss", "split_gauss", "bimodal", "spline")) -> dict:
    """Generate from each parametric family with random params; select by BIC.
    confusion[i, j] = P(select candidate j | generated by family i)."""
    x = probe_x() if x is None else x
    rng = np.random.default_rng(seed)
    gens = {
        "gauss": lambda: dict(mu=rng.uniform(0.5, 1.5), sigma=rng.uniform(0.5, 1.3), amp=rng.uniform(0.7, 1.3), base=rng.uniform(0, 0.2)),
        "split_gauss": lambda: dict(mu=rng.uniform(0.6, 1.4), sl=rng.uniform(0.4, 0.7), sr=rng.uniform(1.1, 1.6), amp=rng.uniform(0.7, 1.3), base=rng.uniform(0, 0.2)),
        "bimodal": lambda: dict(mu1=rng.uniform(-0.2, 0.5), mu2=rng.uniform(1.5, 2.2), sigma=rng.uniform(0.3, 0.5), amp=rng.uniform(0.6, 1.0), base=rng.uniform(0, 0.2)),
    }
    conf = np.zeros((len(gens), len(candidates)))
    for i, (g, draw) in enumerate(gens.items()):
        for _ in range(n_sims):
            y = response(g, draw(), x) + rng.normal(0, noise_sd, len(x))
            bics = []
            for c in candidates:
                try:
                    bics.append(fit(c, x, y)["bic"])
                except RuntimeError:
                    bics.append(np.inf)
            conf[i, int(np.argmin(bics))] += 1
    return {"generators": list(gens), "candidates": list(candidates), "confusion": conf / n_sims, "noise_sd": noise_sd}


def make_figures(out: str | Path, noise_sd: float = 0.1, seed: int = 0) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    files = []
    x, xf = probe_x(), np.linspace(-2.2, 3.8, 300)
    rng = np.random.default_rng(seed)
    base = response(*SCENARIOS["baseline"], xf)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, name in zip(axs.flat, [k for k in SCENARIOS if k != "baseline"]):
        model, p = SCENARIOS[name]
        y = response(model, p, x) + rng.normal(0, noise_sd, len(x))
        ax.plot(xf, base, color="0.7", lw=1, label="baseline")
        ax.plot(xf, response(model, p, xf), "k", lw=1.5, label=f"true ({model})")
        ax.plot(x, y, "o", color="k", ms=3, mfc="w", label="noisy samples")
        fg, fs = fit("gauss", x, y), fit("spline", x, y)
        ax.plot(xf, fg["predict"](xf), "--", color="C0", lw=1, label=f"Gaussian fit (BIC {fg['bic']:.1f})")
        ax.plot(xf, fs["predict"](np.clip(xf, x.min(), x.max())), ":", color="C3", lw=1.2, label=f"spline fit (BIC {fs['bic']:.1f})")
        ax.set_title(name, fontsize=10); ax.legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xticks(np.log2([0.25, 1, 4, 12]), ["0.25", "1", "4", "12"]); ax.set_xlabel("SF (cpd)")
    for ax in axs[:, 0]:
        ax.set_ylabel("response (a.u.)")
    fig.suptitle(f"Simulated tuning scenarios (noise SD {noise_sd})", fontsize=11, color="firebrick")
    fig.tight_layout(); f = out / "synthetic_tuning_scenarios.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)

    pr = parameter_recovery(noise_sd=noise_sd, seed=seed)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    for i, (ax, n) in enumerate(zip(axs, pr["names"])):
        t, r = pr["true"][:, i], pr["recovered"][:, i]
        ax.plot(t, r, "k.", ms=3); lim = [min(t.min(), r.min()), max(t.max(), r.max())]
        ax.plot(lim, lim, color="0.6", lw=0.8)
        s = pr["summary"][n]
        ax.set_title(f"{n}: r={s['r']:.2f}, RMSE={s['rmse']:.2f}", fontsize=9); ax.set_xlabel("true"); ax.set_ylabel("recovered")
    fig.suptitle("Simulated parameter recovery: Gaussian generator, Gaussian fit", fontsize=10, color="firebrick")
    fig.tight_layout(); f = out / "synthetic_parameter_recovery.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)

    mr = model_recovery(noise_sd=noise_sd, seed=seed + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    im = ax.imshow(mr["confusion"], cmap="Greys", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(mr["confusion"]):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v > 0.5 else "k", fontsize=9)
    ax.set_xticks(range(len(mr["candidates"])), mr["candidates"], rotation=20); ax.set_yticks(range(len(mr["generators"])), mr["generators"])
    ax.set_xlabel("selected by BIC"); ax.set_ylabel("generating model"); fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Simulated model recovery", fontsize=10, color="firebrick")
    fig.tight_layout(); f = out / "synthetic_model_recovery.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/simulation")
    ap.add_argument("--noise-sd", type=float, default=0.1)
    a = ap.parse_args(argv)
    for f in make_figures(a.out, a.noise_sd):
        print(f)
    pr = parameter_recovery(noise_sd=a.noise_sd)
    print("Simulated parameter recovery:", {k: {m: round(v, 3) for m, v in d.items()} for k, d in pr["summary"].items()})
    mr = model_recovery(noise_sd=a.noise_sd)
    print("Simulated model recovery (rows=generator, cols=", mr["candidates"], "):\n", mr["confusion"].round(2))


if __name__ == "__main__":
    main()
