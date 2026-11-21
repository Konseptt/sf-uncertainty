import copy
import csv
import json
from collections import Counter
from itertools import pairwise

import numpy as np
import pytest

from sfu import config as cfgmod
from sfu.analysis import load, validate_distributions
from sfu.calibration import Monitor
from sfu.config import ExperimentConfig
from sfu.datalog import COLUMNS
from sfu.distributions import MATCH_TOL, DistSpec, build_conditions, make_grid, pmf, stats
from sfu.responses import SimulatedObserver
from sfu.runner import HeadlessBackend, run_session
from sfu.schedule import balanced_directions, build_session, fixed_counts, williams
from sfu.simulation import fit, gauss, probe_x

GRID = make_grid()


def small_cfg(**design) -> ExperimentConfig:
    d = ExperimentConfig().to_dict()
    d["design"].update({"trials_per_block": 36, "practice_trials": 4, "calibration_block": False, **design})
    return cfgmod.from_dict(d)


# -- distributions -----------------------------------------------------------
@pytest.mark.parametrize("matching", ["mode_matched", "moment_matched"])
def test_pmfs_normalized_nonnegative(matching):
    for name, c in build_conditions(GRID, matching).items():
        assert np.isclose(c["pmf"].sum(), 1.0, atol=1e-12), name
        assert (c["pmf"] >= 0).all(), name


def test_moment_matching_numerically_verified():
    conds = build_conditions(GRID, "moment_matched")
    target = conds["broad"]["stats"]["var_log2"]
    for name in ("broad", "right_skewed", "left_skewed", "bimodal"):
        s, m = conds[name]["stats"], conds[name]["match"]
        assert m["matched"], name
        assert abs(s["mean_log2"] - 1.0) < MATCH_TOL and abs(s["var_log2"] - target) < MATCH_TOL, name
    # uniform has no free parameter -> must be reported unmatched (its variance differs)
    assert conds["uniform"]["match"]["matched"] is False
    # narrow intentionally differs in variance only
    assert abs(conds["narrow"]["stats"]["mean_log2"] - 1.0) < 1e-9
    assert conds["narrow"]["stats"]["var_log2"] < target


def test_infeasible_target_reported_unmatched():
    # variance > max possible on [0,2] grid (=1) cannot be matched
    conds = build_conditions(GRID, "moment_matched", target_sd=1.2)
    assert not conds["broad"]["match"]["matched"]


def test_stats_against_analytic_uniform():
    s = stats(pmf(DistSpec("uniform"), GRID), GRID)
    n, h = len(GRID), GRID[1] - GRID[0]
    assert np.isclose(s["var_log2"], (n**2 - 1) / 12 * h**2)
    assert np.isclose(s["entropy_bits"], np.log2(n))
    assert s["modes_log2"] == [] and abs(s["skewness"]) < 1e-12


def test_skew_direction_and_modes():
    mm = build_conditions(GRID, "mode_matched")
    assert mm["right_skewed"]["stats"]["skewness"] > 0 > mm["left_skewed"]["stats"]["skewness"]
    assert mm["right_skewed"]["stats"]["p_above_center"] > mm["right_skewed"]["stats"]["p_below_center"]
    assert mm["right_skewed"]["stats"]["modes_cpd"] == [2.0]
    assert len(mm["bimodal"]["stats"]["modes_cpd"]) == 2


# -- scheduling -------------------------------------------------------------
def test_fixed_counts_sum_and_rounding():
    for name, c in build_conditions(GRID).items():
        k = fixed_counts(c["pmf"], 108)
        assert k.sum() == 108 and np.all(np.abs(k - c["pmf"] * 108) < 1), name


def test_williams_first_order_balance():
    for n in (4, 5, 6):
        rows = williams(n)
        assert all(sorted(r) == list(range(n)) for r in rows)
        pairs = Counter(p for r in rows for p in pairwise(r))
        assert len(set(pairs.values())) == 1 and len(pairs) == n * (n - 1)


def test_reproducible_and_participant_specific():
    cfg = small_cfg()
    a, b = build_session(cfg, "P01", 1, 0), build_session(cfg, "P01", 1, 0)
    assert json.dumps(a, default=str) == json.dumps(b, default=str)
    c = build_session(cfg, "P02", 1, 0)
    assert a["seed"] != c["seed"]
    assert [t["sf_cpd"] for t in a["blocks"][1]["trials"]] != [t["sf_cpd"] for t in c["blocks"][1]["trials"]]


def test_condition_assignment_and_bounds():
    cfg = small_cfg()
    s = build_session(cfg, "P03", 1, 3)
    main = [b for b in s["blocks"] if b["block_type"] == "main"]
    names = cfg.distributions.conditions
    assert [b["condition"] for b in main] == [names[i] for i in williams(len(names))[3]]
    for b in main:
        p = s["conditions"][b["condition"]]["pmf"]
        for t in b["trials"]:
            assert t["condition"] == b["condition"]
            assert t["p_expected"] == pytest.approx(p[t["level_index"]])
            assert 1.0 - 1e-9 <= t["sf_cpd"] <= 4.0 + 1e-9
        realized = np.bincount([t["level_index"] for t in b["trials"]], minlength=len(p))
        assert (realized == fixed_counts(np.array(p), len(b["trials"]))).all()


def test_response_balance_per_sf():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lv = rng.integers(0, 9, rng.integers(10, 120))
        d = balanced_directions(lv, rng)
        assert abs(d.sum()) <= 1
        for v in np.unique(lv):
            assert abs(d[lv == v].sum()) <= 1


def test_probability_sampling_is_iid_not_exact():
    cfg = small_cfg(sampling="probability", trials_per_block=2000)
    s = build_session(cfg, "P01", 1, 0)
    b = next(b for b in s["blocks"] if b["condition"] == "broad")
    r = np.bincount([t["level_index"] for t in b["trials"]], minlength=9) / 2000
    assert np.abs(r - np.array(b["pmf"])).max() < 0.04


def test_probe_stream_identical_across_conditions():
    s = build_session(small_cfg(paradigm="probe"), "P01", 1, 0)
    streams = [[(t["probe_present"], t["probe_sf_cpd"], t["probe_orientation_deg"]) for t in b["trials"]]
               for b in s["blocks"] if b["block_type"] == "main"]
    assert all(str(x) == str(streams[0]) for x in streams)


# -- calibration / config -------------------------------------------------------
def test_monitor_geometry_and_nyquist():
    m = Monitor(53.1, 57.0, 1920, 1080, 60.0)
    assert m.deg2px(1) == pytest.approx(m.px_per_deg)
    assert m.frames(100) == 6 and m.frames(1) == 1
    with pytest.raises(ValueError):
        Monitor(53.1, 10.0, 1920, 1080, 60.0).check_sf(4.0)
    with pytest.raises(ValueError):
        Monitor(0, 57, 1920, 1080, 60)


def test_config_rejects_bad_values_and_unknown_keys():
    d = ExperimentConfig().to_dict()
    bad = copy.deepcopy(d); bad["stimulus"]["contrast"] = 1.5
    with pytest.raises(ValueError):
        cfgmod.from_dict(bad)
    bad = copy.deepcopy(d); bad["design"]["typo_field"] = 1
    with pytest.raises(ValueError):
        cfgmod.from_dict(bad)
    for section, key, val in (("distributions", "conditions", ["nonexistent"]), ("distributions", "conditions", []),
                              ("design", "trials_per_block", 0), ("design", "blocks_per_condition", 0)):
        bad = copy.deepcopy(d); bad[section][key] = val
        with pytest.raises(ValueError):
            cfgmod.from_dict(bad)


# -- session / logging ------------------------------------------------------------
def _run(cfg, tmp_path, **kw):
    be = HeadlessBackend(cfg, SimulatedObserver(np.random.default_rng(0)))
    return run_session(cfg, "T01", 1, 1, be, tmp_path, **kw)


def test_complete_trial_logging(tmp_path):
    cfg = small_cfg()
    out = _run(cfg, tmp_path)
    rows = list(csv.DictReader(open(out / "trials.csv")))
    n = sum(len(b["trials"]) for b in build_session(cfg, "T01", 1, 1)["blocks"])
    assert len(rows) == n and list(rows[0]) == COLUMNS
    required = [c for c in COLUMNS if not c.startswith("probe_") and c not in ("response", "rt_s", "accuracy")]
    for r in rows:
        assert all(r[c] != "" for c in required)
    side = json.loads((out / "session.json").read_text())
    assert side["status"] == "complete" and side["seed"] == int(rows[0]["seed"])
    assert json.loads((out / "trials.json").read_text())["trials"][0]["trial_uid"] == "0"


def test_abort_and_resume_no_duplicates(tmp_path):
    cfg = small_cfg()
    out = _run(cfg, tmp_path, abort_after=50)
    assert json.loads((out / "session.json").read_text())["status"] == "aborted"
    with pytest.raises(FileExistsError):
        _run(cfg, tmp_path)
    _run(cfg, tmp_path, resume=True)
    runs = json.loads((out / "session.json").read_text())["runs"]
    assert [r["status"] for r in runs] == ["aborted", "complete"]
    assert [r["trials_logged"] for r in runs][0] == 50
    uids = [int(r["trial_uid"]) for r in csv.DictReader(open(out / "trials.csv"))]
    n = sum(len(b["trials"]) for b in build_session(cfg, "T01", 1, 1)["blocks"])
    assert sorted(uids) == list(range(n))


def test_resume_refuses_changed_config(tmp_path):
    _run(small_cfg(), tmp_path, abort_after=5)
    with pytest.raises(RuntimeError):
        _run(small_cfg(cue="text"), tmp_path, resume=True)


def test_calibration_block_writes_frozen_thresholds(tmp_path):
    cfg = small_cfg(calibration_block=True, calibration_trials_per_level=10)
    out = _run(cfg, tmp_path)
    thr = json.loads((out / "thresholds.json").read_text())["tilt_by_level"]
    rows = [r for r in csv.DictReader(open(out / "trials.csv")) if r["block_type"] == "main"]
    for r in rows:
        assert float(r["tilt_deg"]) == pytest.approx(thr[r["level_index"]])


def test_analysis_validation_matches_fixed_counts(tmp_path):
    _run(small_cfg(), tmp_path)
    df, side = load(tmp_path)
    v = validate_distributions(df, side)
    assert len(v) == 6 and (v.max_abs_prob_dev <= 1 / 36 + 1e-12).all()
    assert np.allclose(v.mean_realized, v.mean_intended, atol=0.05)  # whole-trial rounding at 36 trials/block


# -- simulation ---------------------------------------------------------------------
def test_noiseless_gaussian_recovery_exact():
    x = probe_x()
    true = dict(mu=1.3, sigma=0.8, amp=1.2, base=0.05)
    f = fit("gauss", x, gauss(x, **true))
    for k, v in true.items():
        assert f["params"][k] == pytest.approx(v, abs=1e-4)


def test_gui_refuses_export_on_invalid_input(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    from sfu.gui import ResearcherGUI
    g = ResearcherGUI(small_cfg(), tmp_path)
    assert g.export() is not None
    for lab, bad in (("trials/block", "abc"), ("participant", "a b"), ("distance cm", "0")):
        old = g.boxes[lab].text
        g.boxes[lab].set_val(bad)
        assert g.export() is None and g.msg.get_text().startswith("ERROR"), lab
        g.boxes[lab].set_val(old)


def test_wilson_interval_contains_estimate_at_extremes():
    from sfu.analysis import wilson
    for k, n in ((0, 1), (1, 1), (0, 7), (7, 7), (36, 36), (3, 7)):
        lo, hi = wilson(k, n)
        assert 0 <= lo <= k / n <= hi <= 1, (k, n, lo, hi)


def test_position_exact_at_any_polar_angle():
    import math
    from sfu.stimulus import position_deg, position_px
    cfg = small_cfg()
    m = cfg.monitor
    for angle in (0, 45, 90, 135, 200):
        c = copy.deepcopy(cfg)
        c.stimulus.polar_angle_deg = angle
        x, y = position_px(m, position_deg(c))
        assert math.hypot(x, y) == pytest.approx(m.ecc2px(c.stimulus.eccentricity_deg))
        assert math.degrees(math.atan2(y, x)) % 360 == pytest.approx(angle % 360)
    assert position_px(m, (0.0, 0.0)) == (0.0, 0.0)


def test_crash_marks_run_crashed(tmp_path):
    cfg = small_cfg()
    be = HeadlessBackend(cfg, SimulatedObserver(np.random.default_rng(0)))
    calls = {"n": 0}
    real = be.run_trial
    def boom(t, tilt):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(t, tilt)
    be.run_trial = boom
    with pytest.raises(KeyboardInterrupt):
        run_session(cfg, "C01", 1, 0, be, tmp_path)
    side = json.loads((tmp_path / "sub-C01_ses-1" / "session.json").read_text())
    assert side["status"] == "crashed" and side["runs"][0]["trials_logged"] == 2


def test_resume_refuses_csv_with_other_columns(tmp_path):
    out = _run(small_cfg(), tmp_path, abort_after=5)
    rows = (out / "trials.csv").read_text().splitlines()
    (out / "trials.csv").write_text("\n".join([rows[0].replace(",stim_dropped_frames", "")] + rows[1:]) + "\n")
    with pytest.raises(RuntimeError):
        _run(small_cfg(), tmp_path, resume=True)


def test_gui_run_launches_runner_with_exported_config(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import subprocess
    from sfu import gui
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd))
    g = gui.ResearcherGUI(small_cfg(), tmp_path)
    g.boxes["participant"].set_val("R07")
    cmd = g.run()
    assert calls == [cmd]
    cfg_path = cmd[cmd.index("--config") + 1]
    assert cfgmod.load(cfg_path).to_dict() == g.cfg.to_dict()
    assert cmd[cmd.index("--participant") + 1] == "R07" and "--demo" in cmd
    g.r_mark.set_active(1)
    cmd = g.run()
    assert cfgmod.load(cmd[cmd.index("--config") + 1]).design.markers == "lsl"
    g.boxes["trials/block"].set_val("abc")
    assert g.run() is None and len(calls) == 2


def test_bandpass_noise_peaks_at_center_sf_and_is_reproducible():
    from sfu.stimulus import bandpass_noise
    ppd, n = 36.0, 256
    f = np.fft.fftfreq(n, d=1 / ppd)
    r = np.hypot(*np.meshgrid(f, f))
    for center in (0.5, 1.0, 2.0, 4.0, 8.0):
        img = bandpass_noise(n, ppd, center, 0.5, seed=3)
        assert np.abs(img).max() == pytest.approx(1.0)
        power = np.abs(np.fft.fft2(img)) ** 2
        mean_log2 = np.sum(power * np.log2(np.where(r > 0, r, 1))) / power.sum()
        assert abs(mean_log2 - np.log2(center)) < 0.15, center
        assert np.array_equal(img, bandpass_noise(n, ppd, center, 0.5, seed=3))
    assert not np.array_equal(bandpass_noise(n, ppd, 2.0, 0.5, 1), bandpass_noise(n, ppd, 2.0, 0.5, 2))


def test_noise_probe_seeds_identical_across_conditions():
    s = build_session(small_cfg(paradigm="probe", probe_type="noise"), "P01", 1, 0)
    streams = [[(t["probe_present"], t["probe_sf_cpd"], t["probe_noise_seed"], t["probe_type"]) for t in b["trials"]]
               for b in s["blocks"] if b["block_type"] == "main"]
    assert all(x == streams[0] for x in streams)
    assert any(seed >= 0 for _, _, seed, _ in streams[0]) and streams[0][0][3] == "noise"


def test_gui_controls_reach_config_and_realized_check(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from sfu import gui
    g = gui.ResearcherGUI(small_cfg(), tmp_path)
    g.c_conds.set_active(g.cond_names.index("uniform"))          # untick uniform
    g.r_para.set_active(1); g.r_ptype.set_active(0)               # probe paradigm, noise probes
    assert "uniform" not in g.cfg.distributions.conditions
    assert (g.cfg.design.paradigm, g.cfg.design.probe_type) == ("probe", "noise")
    assert "uniform" not in g.sched["order"]
    real = gui.realized_stats(g.sched)
    for name, rs in real.items():                                 # fixed_count: realized matches intended
        assert abs(rs["mean_log2"] - g.sched["conditions"][name]["stats"]["mean_log2"]) <= gui.TOL_OCT
    typ, longest = gui.session_minutes(g.cfg, 100, 4)
    assert 0 < typ <= longest
    lim = gui.display_limits(g.cfg)
    assert lim["sf_hi"] == g.cfg.design.probe_sf_max_cpd and lim["reach_px"] < lim["half_width_px"]
    for name in g.cond_names:                                     # unticking every condition is rejected
        if g.c_conds.get_status()[g.cond_names.index(name)]:
            g.c_conds.set_active(g.cond_names.index(name))
    assert g.export() is None and g.msg.get_text().startswith("ERROR")


def test_gui_fast_redraw_matches_full_redraw(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from sfu import gui
    g = gui.ResearcherGUI(small_cfg(paradigm="probe"), tmp_path)
    g.fig.canvas.draw()
    for v in np.linspace(0.3, 1.2, 8):
        g.sliders["broad_sd"].set_val(v)
    g.r_samp.set_active(1); g.c_conds.set_active(2); g.r_mark.set_active(1); g.boxes["participant"].set_val("Q9")
    fast = np.asarray(g.fig.canvas.buffer_rgba()).copy()
    g.fig.canvas.draw()
    full = np.asarray(g.fig.canvas.buffer_rgba()).copy()
    assert int((np.abs(fast.astype(int) - full.astype(int)).max(axis=2) > 30).sum()) == 0
