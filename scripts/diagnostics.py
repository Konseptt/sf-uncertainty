"""Plots of theoretical vs sampled distributions for both matching schemes and
both sampling methods, plus a statistics table for each scheme.

    PYTHONPATH=. python scripts/diagnostics.py --out figures/diagnostics
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sfu.distributions import build_conditions, make_grid, stats, stats_table
from sfu.schedule import sample_levels

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figures/diagnostics")
ap.add_argument("--n-trials", type=int, default=108)
ap.add_argument("--n-blocks", type=int, default=200, help="simulated blocks for sampling-error spread")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
g = make_grid()
rng = np.random.default_rng(0)
for mode in ("moment_matched", "mode_matched"):
    conds = build_conditions(g, mode)
    (out / f"stats_{mode}.txt").write_text(stats_table(conds) + "\n")
    json.dump({k: {"pmf": v["pmf"].tolist(), "stats": v["stats"], "match": v["match"], "spec": v["spec"].to_dict()}
               for k, v in conds.items()}, open(out / f"conditions_{mode}.json", "w"), indent=2)
    fig, axs = plt.subplots(2, 6, figsize=(17, 5.5), sharey="row")
    for j, (name, c) in enumerate(conds.items()):
        p = c["pmf"]
        for i, method in enumerate(("probability", "fixed_count")):
            ax = axs[i, j]
            R = np.array([np.bincount(sample_levels(p, a.n_trials, method, rng), minlength=len(g)) / a.n_trials
                          for _ in range(a.n_blocks)])
            ax.bar(g, p, width=0.2, color="0.8", label="theoretical")
            med = np.median(R, 0)
            ax.errorbar(g, med, yerr=[med - np.quantile(R, .025, 0), np.quantile(R, .975, 0) - med],
                        fmt="ko", ms=3, lw=1, label=f"realized median, 95% range over {a.n_blocks} blocks")
            sd_err = np.array([stats(r, g)["sd_octaves"] for r in R]) - c["stats"]["sd_octaves"]
            ax.set_title(f"{name}\n{method}: SD err {sd_err.mean():+.3f} [{np.quantile(sd_err,.025):+.3f},{np.quantile(sd_err,.975):+.3f}] oct", fontsize=8)
            ax.set_xticks(g[::2], [f"{2**v:.2g}" for v in g[::2]], fontsize=7)
    axs[0, 0].legend(fontsize=6, frameon=False); axs[1, 0].set_xlabel("SF (cpd)")
    fig.suptitle(f"{mode}: theoretical vs sampled ({a.n_trials} trials/block)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"theoretical_vs_sampled_{mode}.png", dpi=120); plt.close(fig)
    print(f"== {mode} ==\n{stats_table(conds)}\n")
print("wrote", out)
