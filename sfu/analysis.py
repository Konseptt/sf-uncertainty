"""Behavioral analysis + stimulus-distribution validation.

    python -m sfu.analysis data/demo --out figures/demo

Behavioral accuracy/RT differences across expectation conditions are NOT
evidence about neural SF tuning. They can arise from decision-level priors,
response bias, attention, or adaptation. Tuning claims need neural data
(e.g. fMRI pSFT fits with the pSF-Toolbox) and the controls in docs/PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .distributions import stats as pmf_stats


def load(root: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load every sub-*/trials.csv below root. Returns (trials, {session_dir: sidecar})."""
    root = Path(root)
    paths = sorted(root.rglob("trials.csv"))
    if not paths:
        raise FileNotFoundError(f"no trials.csv under {root}")
    dfs, side = [], {}
    for p in paths:
        df = pd.read_csv(p)
        df["session_dir"] = p.parent.name
        dfs.append(df)
        sj = p.parent / "session.json"
        side[p.parent.name] = json.loads(sj.read_text()) if sj.exists() else {}
    return pd.concat(dfs, ignore_index=True), side


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (min(c - h, p), max(c + h, p))  # float rounding can put a bound inside p at p=0 or 1


def bootstrap_ci(x, fn=np.median, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = fn(x[rng.integers(0, len(x), (n_boot, len(x)))], axis=1)
    return tuple(np.quantile(b, [alpha / 2, 1 - alpha / 2]))


def validate_distributions(df: pd.DataFrame, side: dict) -> pd.DataFrame:
    """Per main block: intended vs realized moments; chi-square GOF (informative only
    for probability sampling - fixed_count matches by construction up to rounding)."""
    rows = []
    for (sd, b), g in df[df.block_type == "main"].groupby(["session_dir", "block_index"]):
        cond = g.condition.iloc[0]
        meta = side.get(sd, {}).get("conditions", {}).get(cond)
        if not meta:
            continue
        p, grid = np.array(meta["pmf"]), np.array(side[sd]["grid_log2"])
        counts = np.bincount(g.level_index.astype(int), minlength=len(p))
        r = counts / counts.sum()
        si, sr = pmf_stats(p, grid), pmf_stats(r, grid)
        exp = p * counts.sum()
        mask = exp > 0
        chi2, pval = sps.chisquare(counts[mask], exp[mask] * counts[mask].sum() / exp[mask].sum()) if mask.sum() > 1 else (np.nan, np.nan)
        rows.append(dict(session_dir=sd, block_index=b, condition=cond, sampling=g.sampling.iloc[0], n=int(counts.sum()),
                         mean_intended=si["mean_log2"], mean_realized=sr["mean_log2"],
                         sd_intended=si["sd_octaves"], sd_realized=sr["sd_octaves"],
                         skew_intended=si["skewness"], skew_realized=sr["skewness"],
                         entropy_intended=si["entropy_bits"], entropy_realized=sr["entropy_bits"],
                         max_abs_prob_dev=float(np.abs(r - p).max()), chi2=chi2, chi2_p=pval))
    return pd.DataFrame(rows)


def behavior_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (Wilson 95% CI) and median correct RT (bootstrap 95% CI) by condition x SF."""
    d = df[(df.block_type == "main") & df.accuracy.notna()]
    rows = []
    for (c, sf), g in d.groupby(["condition", "sf_cpd"]):
        k, n = g.accuracy.sum(), len(g)
        lo, hi = wilson(k, n)
        rt = g.loc[g.accuracy == 1, "rt_s"]
        rlo, rhi = bootstrap_ci(rt)
        rows.append(dict(condition=c, sf_cpd=sf, n=n, accuracy=k / n, acc_lo=lo, acc_hi=hi,
                         rt_median=rt.median(), rt_lo=rlo, rt_hi=rhi, mean_p_expected=g.p_expected.mean()))
    return pd.DataFrame(rows)


def plot_all(df: pd.DataFrame, side: dict, out: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    synthetic = bool(df.get("synthetic", pd.Series([False])).astype(str).eq("True").any())
    tag = "  [SYNTHETIC DATA - simulated observer]" if synthetic else ""
    files = []
    # 1. intended vs realized
    main = df[df.block_type == "main"]
    conds = list(dict.fromkeys(main.condition))
    fig, axs = plt.subplots(1, len(conds), figsize=(2.6 * len(conds), 2.8), sharey=True, squeeze=False)
    for ax, c in zip(axs[0], conds):
        g = main[main.condition == c]
        sd0 = g.session_dir.iloc[0]
        p = np.array(side[sd0]["conditions"][c]["pmf"])
        grid = np.array(side[sd0]["grid_log2"])
        r = np.bincount(g.level_index.astype(int), minlength=len(p)) / len(g)
        ax.bar(grid, p, width=0.18, color="0.75", label="intended")
        ax.plot(grid, r, "ko", ms=3.5, label=f"realized (n={len(g)})")
        ax.set_title(c, fontsize=9); ax.set_xticks(grid[::2], [f"{2**v:.2g}" for v in grid[::2]])
        ax.set_xlabel("SF (cpd)")
    axs[0][0].set_ylabel("probability"); axs[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Intended vs realized SF distributions (pooled sessions)" + tag, fontsize=10)
    fig.tight_layout(); f = out / "distributions_intended_vs_realized.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)
    # 2. behavior
    bt = behavior_table(df)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (c, g) in enumerate(bt.groupby("condition")):
        g = g.sort_values("sf_cpd"); x = np.log2(g.sf_cpd) + (i - 2.5) * 0.015
        a1.errorbar(x, g.accuracy, [g.accuracy - g.acc_lo, g.acc_hi - g.accuracy], marker="o", ms=3, lw=1, capsize=0, label=c)
        a2.errorbar(x, g.rt_median, [g.rt_median - g.rt_lo, g.rt_hi - g.rt_median], marker="o", ms=3, lw=1, capsize=0)
    for a, yl in ((a1, "proportion correct (95% Wilson CI)"), (a2, "median correct RT, s (95% bootstrap CI)")):
        ticks = np.log2(np.sort(bt.sf_cpd.unique()))
        a.set_xticks(ticks[::2], [f"{2**t:.2g}" for t in ticks[::2]]); a.set_xlabel("SF (cpd)"); a.set_ylabel(yl)
    a1.axhline(0.5, color="0.6", lw=0.8, ls=":"); a1.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Orientation discrimination by SF and expectation condition" + tag, fontsize=10)
    fig.tight_layout(); f = out / "behavior_by_sf_condition.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)
    # 3. timing QC
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3))
    a1.hist(df.duration_measured_ms.dropna(), bins=30, color="0.4"); a1.set_xlabel("measured stimulus duration (ms)")
    col = "stim_dropped_frames" if "stim_dropped_frames" in df else "dropped_frames"
    a2.hist(df[col], bins=np.arange(df[col].max() + 2) - 0.5, color="0.4"); a2.set_xlabel(col.replace("_", " ") + " per trial")
    fig.suptitle("Timing QC" + tag, fontsize=10); fig.tight_layout()
    f = out / "timing_qc.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    df, side = load(a.root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    v = validate_distributions(df, side); v.to_csv(out / "distribution_validation.csv", index=False)
    b = behavior_table(df); b.to_csv(out / "behavior_summary.csv", index=False)
    files = plot_all(df, side, out)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(v.round(4).to_string(index=False)); print(); print(b.round(3).to_string(index=False))
    print("\nNOTE: behavioral effects are not direct evidence of neural tuning changes.")
    if df.synthetic.astype(str).eq("True").any():
        print("NOTE: data contain SYNTHETIC (simulated-observer) trials.")
    print("\n".join(map(str, files)))


if __name__ == "__main__":
    main()
