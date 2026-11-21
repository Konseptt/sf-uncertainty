"""Trial scheduling: seeds, counterbalancing, SF sampling, response balancing
and probes.

Sampling (design.sampling):
* probability: each trial is an independent draw from the PMF, so realized
  proportions vary by block and must be reported per block.
* fixed_count: counts per level are set by largest-remainder rounding and then
  shuffled. Realized proportions match the PMF, but late trials in a block are
  in principle predictable.

All randomness comes from np.random.SeedSequence([study_seed, hash(participant,
session)]), with one child seed per block so any block can be regenerated on its
own when resuming.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .config import ExperimentConfig
from .distributions import DistSpec, build_conditions, make_grid, pmf, stats
from .stimulus import envelope_sd_deg, position_deg

SYMBOLS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def master_seed(study_seed: int, participant: str, session: int) -> int:
    h = int.from_bytes(hashlib.sha256(f"{participant}|{session}".encode()).digest()[:8], "big")
    return int(np.random.SeedSequence([study_seed, h]).generate_state(1, np.uint64)[0])


def williams(n: int) -> list[list[int]]:
    """Williams balanced Latin square: each ordered adjacent pair appears equally often.
    n even -> n rows; n odd -> 2n rows (square + its mirror)."""
    first = [0]
    lo, hi = 1, n - 1
    for k in range(1, n):
        if k % 2:
            first.append(lo); lo += 1
        else:
            first.append(hi); hi -= 1
    rows = [[(x + i) % n for x in first] for i in range(n)]
    return rows if n % 2 == 0 else rows + [r[::-1] for r in rows]


def fixed_counts(p: np.ndarray, n: int) -> np.ndarray:
    """Largest-remainder (Hamilton) rounding of p*n to integers summing to n."""
    raw = np.asarray(p) * n
    c = np.floor(raw).astype(int)
    rem = n - c.sum()
    order = np.argsort(-(raw - c), kind="stable")
    c[order[:rem]] += 1
    return c


def sample_levels(p: np.ndarray, n: int, method: str, rng: np.random.Generator) -> np.ndarray:
    if method == "probability":
        return rng.choice(len(p), size=n, p=p)
    if method == "fixed_count":
        return rng.permutation(np.repeat(np.arange(len(p)), fixed_counts(p, n)))
    raise ValueError(f"unknown sampling {method!r}")


def balanced_directions(levels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """+1 (clockwise) / -1 (counter-clockwise) balanced within each SF level
    (difference <= 1 per level; odd remainders alternate across levels so the
    block total is also balanced to within 1)."""
    d = np.zeros(len(levels), int)
    flip = int(rng.integers(2))
    for lv in np.unique(levels):
        idx = np.flatnonzero(levels == lv)
        k = len(idx)
        base = np.array([1, -1] * (k // 2) + ([1 - 2 * flip] if k % 2 else []), int)
        flip ^= k % 2
        d[idx] = rng.permutation(base)
    return d


def condition_set(cfg: ExperimentConfig) -> tuple[np.ndarray, dict]:
    dc = cfg.distributions
    grid = make_grid(dc.sf_min_cpd, dc.sf_max_cpd, dc.n_levels)
    center = float(np.mean(grid[[0, -1]]))
    conds = build_conditions(grid, dc.matching, center, dc.narrow_sd, dc.broad_sd, dc.skew_alpha,
                             dc.bimodal_sep, dc.bimodal_sd, dc.target_sd)
    for name, d in dc.custom.items():
        spec = DistSpec(d["family"], d.get("params", {}))
        p = pmf(spec, grid)
        conds[name] = {"spec": spec, "pmf": p, "stats": stats(p, grid, center), "match": None}
    return grid, conds


def block_order(cfg: ExperimentConfig, cb_index: int, rng: np.random.Generator) -> list[str]:
    names = list(cfg.distributions.conditions)
    mode = cfg.design.counterbalance
    if mode == "williams":
        rows = williams(len(names))
        order = [names[i] for i in rows[cb_index % len(rows)]]
    elif mode == "random":
        order = [names[i] for i in rng.permutation(len(names))]
    else:
        order = names
    out: list[str] = []
    for r in range(cfg.design.blocks_per_condition):  # ABC CBA ABC ... reduces linear drift
        out += order if r % 2 == 0 else order[::-1]
    return out


def _block(cfg, grid, name, cond, n, method, seed, meta, probe_seed=None) -> list[dict]:
    rng = np.random.default_rng(seed)
    s, dsn = cfg.stimulus, cfg.design
    p = cond["pmf"]
    levels = sample_levels(p, n, method, rng)
    dirs = balanced_directions(levels, rng)
    phases = rng.random(n) if s.randomize_phase else np.zeros(n)
    itis = rng.uniform(*s.iti_ms, size=n)
    realized = np.bincount(levels, minlength=len(grid)) / n
    pos = position_deg(cfg)
    probe = None
    if dsn.paradigm == "probe" and probe_seed is not None:
        # identical probe stream for every condition with the same probe_seed
        prng = np.random.default_rng(probe_seed)
        k = int(round(dsn.probe_rate * n))
        present = np.zeros(n, bool)
        present[prng.permutation(n)[:k]] = True
        pgrid = make_grid(dsn.probe_sf_min_cpd, dsn.probe_sf_max_cpd, dsn.probe_n_levels)
        plev = prng.permutation(np.repeat(np.arange(len(pgrid)), fixed_counts(np.full(len(pgrid), 1 / len(pgrid)), k)))
        pori, pph = prng.uniform(0, 180, k), prng.random(k)
        pseeds = prng.integers(0, 2**31 - 1, k)
        probe = (present, plev, pori, pph, position_deg(cfg, mirror=True), pgrid, pseeds)
    trials, run_counts, j = [], np.zeros(len(grid)), 0
    for t in range(n):
        lv = int(levels[t])
        run_counts[lv] += 1
        sf = float(2 ** grid[lv])
        tr = dict(meta, trial_index=t, condition=name,
                  dist_family=cond["spec"].family, dist_params=json.dumps(cond["spec"].to_dict()["params"]),
                  level_index=lv, sf_log2=float(grid[lv]), sf_cpd=sf, tilt_dir=int(dirs[t]),
                  correct_key=cfg.key_right if dirs[t] > 0 else cfg.key_left,
                  phase=float(phases[t]), contrast=s.contrast, envelope_sd_deg=envelope_sd_deg(cfg, sf),
                  pos_x_deg=pos[0], pos_y_deg=pos[1], ecc_deg=s.eccentricity_deg, iti_ms=float(itis[t]),
                  p_expected=float(p[lv]), p_observed_block=float(realized[lv]),
                  p_observed_running=float(run_counts[lv] / (t + 1)), block_seed=int(seed),
                  probe_present=False, probe_sf_cpd=np.nan, probe_sf_log2=np.nan, probe_orientation_deg=np.nan,
                  probe_phase=np.nan, probe_pos_x_deg=np.nan, probe_pos_y_deg=np.nan,
                  probe_type=dsn.probe_type if dsn.paradigm == "probe" else "", probe_noise_seed=-1)
        if probe is not None and probe[0][t]:
            pl = int(probe[1][j])
            pg = probe[5]
            tr.update(probe_present=True, probe_sf_log2=float(pg[pl]), probe_sf_cpd=float(2 ** pg[pl]),
                      probe_orientation_deg=float(probe[2][j]), probe_phase=float(probe[3][j]),
                      probe_pos_x_deg=probe[4][0], probe_pos_y_deg=probe[4][1], probe_noise_seed=int(probe[6][j]))
            j += 1
        trials.append(tr)
    return trials


def build_session(cfg: ExperimentConfig, participant: str, session: int, cb_index: int) -> dict:
    """Deterministic full-session schedule. Same inputs -> identical output."""
    seed = master_seed(cfg.study_seed, participant, session)
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss.spawn(1)[0])
    grid, conds = condition_set(cfg)
    order = block_order(cfg, cb_index, rng)
    symbols = dict(zip(sorted(conds), rng.permutation(list(SYMBOLS[:len(conds)]))))
    d = cfg.design
    blocks, uid = [], 0
    uniform = {"spec": DistSpec("uniform"), "pmf": pmf(DistSpec("uniform"), grid)}
    plan = []
    if d.practice_trials:
        plan.append(("practice", "uniform", uniform, d.practice_trials, "fixed_count"))
    if d.calibration_block:
        plan.append(("calibration", "uniform", uniform, d.calibration_trials_per_level * len(grid), "fixed_count"))
    rep_seen: dict[str, int] = {}
    for name in order:
        plan.append(("main", name, conds[name], d.trials_per_block, d.sampling))
    child = ss.spawn(len(plan) + 1)[1:]
    for b, (btype, name, cond, n, method) in enumerate(plan):
        rep = rep_seen.get(f"{btype}:{name}", 0)
        rep_seen[f"{btype}:{name}"] = rep + 1
        block_seed = int(child[b].generate_state(1, np.uint64)[0])
        pseed = int(np.random.SeedSequence([seed, 7919, rep]).generate_state(1, np.uint64)[0]) if btype == "main" else None
        meta = dict(participant=participant, session=session, cb_index=cb_index, block_index=b,
                    block_type=btype, block_rep=rep, matching=cfg.distributions.matching,
                    sampling=method, cue=d.cue, seed=seed)
        trials = _block(cfg, grid, name, cond, n, method, block_seed, meta, pseed)
        for t in trials:
            t["trial_uid"] = uid; uid += 1
        blocks.append({"block_index": b, "block_type": btype, "condition": name, "trials": trials,
                       "pmf": cond["pmf"].tolist(), "symbol": symbols.get(name, "")})
    return {"seed": seed, "grid_log2": grid.tolist(), "order": order, "symbols": symbols,
            "conditions": {k: {"spec": v["spec"].to_dict(), "pmf": v["pmf"].tolist(), "stats": v.get("stats"),
                               "match": v.get("match")} for k, v in conds.items()},
            "blocks": blocks}
