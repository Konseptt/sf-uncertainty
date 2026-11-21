"""Researcher interface: choose conditions, paradigm and distributions, set monitor
and timing values, check display limits and realized trial statistics, preview
stimuli at the monitor's pixels per degree, and export or run a session.

    python -m sfu.gui --config configs/demo.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import config as cfgmod
from .config import ExperimentConfig, max_envelope_sd_deg
from .datalog import COLUMNS, write_json
from .distributions import CONDITIONS, stats as pmf_stats
from .schedule import build_session
from .stimulus import noise_kwargs, render_gabor

FIELDS = [  # (label, config path or None for session ids, type)
    ("participant", None, str), ("session", None, int), ("cb index", None, int),
    ("trials/block", "design.trials_per_block", int), ("study seed", "study_seed", int),
    ("distance cm", "monitor.distance_cm", float), ("width cm", "monitor.width_cm", float),
    ("width px", "monitor.width_px", int), ("refresh Hz", "monitor.refresh_hz", float),
    ("stim ms", "stimulus.duration_ms", float), ("contrast", "stimulus.contrast", float),
    ("ecc deg", "stimulus.eccentricity_deg", float),
]
SLIDERS = [("narrow_sd", 0.05, 0.6), ("broad_sd", 0.2, 1.5), ("skew_alpha", 0.0, 15.0),
           ("bimodal_sd", 0.05, 0.4), ("bimodal_sep", 0.4, 1.8)]
TOL_OCT = 0.05          # realized mean or SD further than this from intended is flagged
TYPICAL_RT_S = 0.7      # used only for the session length estimate
BREAK_S = 20            # assumed time per block intro and break


def _get(cfg, path):
    o = cfg
    for k in path.split("."):
        o = getattr(o, k)
    return o


def _set(cfg: ExperimentConfig, path: str, val) -> ExperimentConfig:
    head, _, leaf = path.rpartition(".")
    if not head:
        return replace(cfg, **{leaf: val})
    return replace(cfg, **{head: replace(getattr(cfg, head), **{leaf: val})})


def _radio(fig, rect, labels, current, RadioButtons):
    return RadioButtons(fig.add_axes(rect), labels, active=labels.index(current) if current in labels else 0)


def session_minutes(cfg: ExperimentConfig, n_trials: int, n_blocks: int) -> tuple[float, float]:
    """(typical, maximum) session length in minutes."""
    s = cfg.stimulus
    base = s.fixation_ms / 1000 + np.mean(s.iti_ms) / 1000
    breaks = n_blocks * BREAK_S
    typical = n_trials * (base + max(TYPICAL_RT_S, s.duration_ms / 1000)) + breaks
    longest = n_trials * (base + s.response_window_ms / 1000) + breaks
    return typical / 60, longest / 60


def display_limits(cfg: ExperimentConfig) -> dict:
    m, d = cfg.monitor, cfg.design
    sf_hi = max(cfg.distributions.sf_max_cpd, d.probe_sf_max_cpd if d.paradigm == "probe" else 0)
    reach = abs(m.ecc2px(cfg.stimulus.eccentricity_deg)) + m.deg2px(6 * max_envelope_sd_deg(cfg)) / 2
    return {"px_per_deg": m.px_per_deg, "sf_hi": sf_hi, "px_per_cycle": m.px_per_deg / sf_hi,
            "min_px_per_cycle": cfg.min_px_per_cycle, "reach_px": reach, "half_width_px": m.width_px / 2}


def realized_stats(sched: dict) -> dict[str, dict]:
    """Statistics of the trials actually generated, pooled over each condition's main blocks."""
    g = np.array(sched["grid_log2"])
    counts: dict[str, np.ndarray] = {}
    for b in sched["blocks"]:
        if b["block_type"] == "main":
            c = np.bincount([t["level_index"] for t in b["trials"]], minlength=len(g))
            counts[b["condition"]] = counts.get(b["condition"], 0) + c
    return {k: pmf_stats(v / v.sum(), g, float(np.mean(g[[0, -1]]))) | {"pmf": v / v.sum()} for k, v in counts.items()}


class ResearcherGUI:
    def __init__(self, cfg: ExperimentConfig, export_dir: str = "exports"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
        self.cfg, self.export_dir = copy.deepcopy(cfg), Path(export_dir)
        self.ids = {"participant": "P01", "session": 1, "cb index": 0}
        self.sched = None
        self.fig = fig = plt.figure(figsize=(16, 9.5))
        fig.canvas.manager and fig.canvas.manager.set_window_title("SF uncertainty - researcher interface")
        x0 = 0.015
        d = cfg.distributions

        fig.text(x0, 0.965, "Distributions", weight="bold")
        self.r_match = _radio(fig, [x0, 0.885, 0.12, 0.07], ["mode_matched", "moment_matched"], d.matching, RadioButtons)
        self.r_samp = _radio(fig, [x0 + 0.125, 0.885, 0.10, 0.07], ["fixed_count", "probability"], cfg.design.sampling, RadioButtons)
        # Backdrop behind the sliders, labels and values, repainted and blitted on its own while dragging.
        self.ax_spanel = fig.add_axes([x0, 0.845 - (len(SLIDERS) - 1) * 0.03 - 0.006, 0.25, len(SLIDERS) * 0.03],
                                      facecolor=fig.get_facecolor())
        self.ax_spanel.set_xticks([]); self.ax_spanel.set_yticks([])
        for sp in self.ax_spanel.spines.values():
            sp.set_visible(False)
        self.sliders = {}
        for i, (k, lo, hi) in enumerate(SLIDERS):
            ax = fig.add_axes([x0 + 0.07, 0.845 - i * 0.03, 0.14, 0.02])
            self.sliders[k] = Slider(ax, k, lo, hi, valinit=getattr(d, k))
            self.sliders[k].drawon = False  # no full-figure redraw per mouse move
            self.sliders[k].on_changed(lambda _v: (self._blit_sliders(), self._refresh_later()))

        self.cond_names = list(dict.fromkeys(list(CONDITIONS) + list(d.custom)))
        fig.text(x0, 0.70, "Conditions run", fontsize=9)
        self.c_conds = CheckButtons(fig.add_axes([x0, 0.575, 0.12, 0.12]), self.cond_names,
                                    [n in d.conditions for n in self.cond_names])
        fig.text(x0 + 0.125, 0.70, "Paradigm", fontsize=9)
        self.r_para = _radio(fig, [x0 + 0.125, 0.64, 0.10, 0.055], ["orientation", "probe"], cfg.design.paradigm, RadioButtons)
        fig.text(x0 + 0.125, 0.628, "Probe type", fontsize=9)
        self.r_ptype = _radio(fig, [x0 + 0.125, 0.575, 0.10, 0.05], ["noise", "gabor"], cfg.design.probe_type, RadioButtons)

        fig.text(x0, 0.545, "Session / monitor / timing", weight="bold")
        self.boxes = {}
        for i, (lab, path, _t) in enumerate(FIELDS):
            ax = fig.add_axes([x0 + 0.07 + (i % 2) * 0.115, 0.505 - (i // 2) * 0.032, 0.045, 0.025])
            self.boxes[lab] = TextBox(ax, lab + " ", initial=str(self.ids[lab] if path is None else _get(cfg, path)))
            self.boxes[lab].on_submit(lambda _t: self.refresh())
        fig.text(x0, 0.31, "EEG markers", fontsize=9)
        self.r_mark = _radio(fig, [x0 + 0.07, 0.29, 0.16, 0.04], ["none", "lsl"],
                             cfg.design.markers if cfg.design.markers in ("none", "lsl") else "none", RadioButtons)
        self.b_gen = Button(fig.add_axes([x0, 0.235, 0.11, 0.04]), "Generate sequence")
        self.b_exp = Button(fig.add_axes([x0 + 0.12, 0.235, 0.11, 0.04]), "Export")
        self.b_run = Button(fig.add_axes([x0, 0.185, 0.23, 0.04]), "Run session with these settings")
        self.b_gen.on_clicked(lambda _e: self.generate())
        self.b_exp.on_clicked(lambda _e: self.export())
        self.b_run.on_clicked(lambda _e: self.run())
        for w in (self.r_match, self.r_samp, self.r_para, self.r_ptype, self.r_mark, self.c_conds):
            w.on_clicked(lambda _l: self.refresh())
        self.msg = fig.text(x0, 0.165, "", fontsize=8, va="top", family="monospace")

        # Right side: status banner, distribution plots, stimulus previews, statistics table.
        self.banner = fig.text(0.29, 0.965, "", fontsize=10, weight="bold", va="center")
        self.limits = fig.text(0.29, 0.935, "", fontsize=8, va="center", family="monospace")
        self.ax_d, self.bars, self.dots = [], [], []
        for i in range(6):
            ax = fig.add_axes([0.29 + (i % 3) * 0.155, 0.605 - (i // 3) * 0.265, 0.135, 0.2])
            self.ax_d.append(ax)
            self.bars.append(ax.bar(np.arange(9), np.zeros(9), width=0.8, color="0.75", label="intended"))
            self.dots.append(ax.plot([], [], "o", ms=4, color="k", label="realized")[0])
            ax.set_xlabel("SF (cpd)", fontsize=7); ax.tick_params(labelsize=7)
        self.ax_d[0].legend(fontsize=7, frameon=False)
        self.ax_g, self.imgs = [], []
        for i, title in enumerate(("target", "target", "target", "probe")):
            ax = fig.add_axes([0.765 + i * 0.056, 0.71, 0.052, 0.17])
            self.imgs.append(ax.imshow(np.zeros((2, 2)), cmap="gray", vmin=-1, vmax=1, interpolation="nearest"))
            ax.tick_params(labelsize=6); self.ax_g.append(ax)
        self._preview_key = None
        self.ax_t = fig.add_axes([0.29, 0.02, 0.69, 0.25]); self.ax_t.axis("off")
        self.cols = ["condition", "runs", "mean cpd int/real", "SD oct int/real", "skew int/real",
                     "entropy bits", "P(>2cpd)", "modes cpd", "matched", "realized check"]
        self.n_rows = len(self.cond_names)
        self.table = self.ax_t.table(cellText=[[""] * len(self.cols)] * self.n_rows, colLabels=self.cols,
                                     loc="upper center", cellLoc="center")
        self.table.auto_set_font_size(False); self.table.set_fontsize(8); self.table.scale(1, 1.25)
        self.order_text = self.ax_t.text(0, -0.02, "", fontsize=8, transform=self.ax_t.transAxes)

        self._timer = fig.canvas.new_timer(interval=150)  # redraw once a slider drag pauses
        self._timer.single_shot = True
        self._timer.add_callback(self.refresh)
        self.refresh()

    def _blit_sliders(self):
        c = self.fig.canvas
        if not c.supports_blit:
            c.draw_idle()
            return
        self.fig.draw_artist(self.ax_spanel)
        for sl in self.sliders.values():
            self.fig.draw_artist(sl.ax)
        c.blit(self.ax_spanel.bbox)

    def _refresh_later(self):
        self._timer.stop()
        self._timer.start()

    # -- state ----------------------------------------------------------
    def _read(self) -> ExperimentConfig:
        c = self.cfg
        on = self.c_conds.get_status()
        dist = replace(c.distributions, matching=self.r_match.value_selected,
                       conditions=[n for n, a in zip(self.cond_names, on) if a],
                       **{k: float(s.val) for k, s in self.sliders.items()})
        design = replace(c.design, sampling=self.r_samp.value_selected, markers=self.r_mark.value_selected,
                         paradigm=self.r_para.value_selected, probe_type=self.r_ptype.value_selected)
        c = replace(c, distributions=dist, design=design)
        for lab, path, typ in FIELDS:
            v = typ(self.boxes[lab].text)
            if path is None:
                self.ids[lab] = v
            else:
                c = _set(c, path, v)
        return c.validate()

    def refresh(self) -> bool:
        """Rebuild the schedule from the widgets. False (and an ERROR line) if any input is invalid."""
        try:
            self.cfg = self._read()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", str(self.ids["participant"])):
                raise ValueError("participant id: letters, digits, _ and - only")
            self.sched = build_session(self.cfg, str(self.ids["participant"]), int(self.ids["session"]), int(self.ids["cb index"]))
            self.msg.set_text("")
        except (ValueError, KeyError) as e:
            self.msg.set_text(f"ERROR: {e}")
            self.banner.set_text("Settings invalid; nothing will be exported or run"); self.banner.set_color("firebrick")
            self.fig.canvas.draw_idle()
            return False
        self._draw()
        return True

    def generate(self):
        if self.refresh():
            n = sum(len(b["trials"]) for b in self.sched["blocks"])
            self.msg.set_text(f"seed {self.sched['seed']}\n{n} trials\norder: {' > '.join(self.sched['order'])}")
            self.fig.canvas.draw_idle()

    # -- drawing (artists are created once and updated in place) ---------
    def _draw(self):
        s, c = self.sched, self.cfg
        g = np.array(s["grid_log2"])
        real = realized_stats(s)
        runs = {n: s["order"].count(n) for n in s["conditions"]}

        n_trials = sum(len(b["trials"]) for b in s["blocks"])
        typ, longest = session_minutes(c, n_trials, len(s["blocks"]))
        lim = display_limits(c)
        cal = "monitor calibrated" if c.monitor.calibrated else "MONITOR NOT CALIBRATED: cpd values are nominal, demo only"
        self.banner.set_text(f"{cal}   |   {n_trials} trials, about {typ:.0f} min (at most {longest:.0f})")
        self.banner.set_color("darkgreen" if c.monitor.calibrated else "firebrick")
        self.limits.set_text(f"{lim['px_per_deg']:.1f} px/deg   {lim['sf_hi']:g} cpd = {lim['px_per_cycle']:.1f} px/cycle "
                             f"(min {lim['min_px_per_cycle']:g})   stimulus edge {lim['reach_px']:.0f} of "
                             f"{lim['half_width_px']:.0f} px from centre   paradigm: {c.design.paradigm}")

        if getattr(self, "_grid", None) is None or not np.array_equal(self._grid, g):  # ticks only change with the grid
            self._grid = g
            for ax in self.ax_d:
                ax.set_xticks(np.arange(len(g))[::2], [f"{2**v:.2g}" for v in g[::2]])
        for ax, bars, dots, name in zip(self.ax_d, self.bars, self.dots, self.cond_names):
            cond = s["conditions"].get(name)
            p = np.array(cond["pmf"]) if cond else np.zeros(len(g))
            for rect, h in zip(bars, p):
                rect.set_height(h)
            r = real.get(name)
            dots.set_data(np.arange(len(g)), r["pmf"]) if r else dots.set_data([], [])
            m = (cond or {}).get("match") or {}
            flag = "" if "matched" not in m else ("  [matched]" if m["matched"] else "  [not matched]")
            ax.set_title(name + flag + ("" if name in runs and runs[name] else "  (not run)"), fontsize=9,
                         color="k" if runs.get(name) else "0.6")
            ax.set_ylim(0, max(0.5, p.max() * 1.15))

        key = (c.monitor, c.stimulus, c.distributions.sf_min_cpd, c.distributions.sf_max_cpd,
               c.design.paradigm, c.design.probe_type, c.design.probe_noise_bandwidth_oct, c.design.probe_contrast)
        if key != self._preview_key:  # images depend only on these settings
            self._preview_key = key
            field = 6 * max_envelope_sd_deg(c)
            for ax, im, sf in zip(self.ax_g[:3], self.imgs[:3], (2 ** g[0], 2 ** g[len(g) // 2], 2 ** g[-1])):
                img, ext = render_gabor(c, sf, ori=0, phase=0.25, field_deg=field)
                im.set_data(img); im.set_extent([-ext / 2, ext / 2, -ext / 2, ext / 2])
                ax.set_title(f"target {sf:.2f} cpd", fontsize=7)
            ax, im = self.ax_g[3], self.imgs[3]
            if c.design.paradigm == "probe":
                sf = float(np.sqrt(c.design.probe_sf_min_cpd * c.design.probe_sf_max_cpd))
                if c.design.probe_type == "noise":
                    img = noise_kwargs(c, sf, 1, (0, 0))["image"] * c.design.probe_contrast
                    n = img.shape[0]; y, x = np.mgrid[:n, :n] - (n - 1) / 2
                    img = img * np.exp(-(x**2 + y**2) / (2 * (n / 6) ** 2)); ext = n / c.monitor.px_per_deg
                else:
                    img, ext = render_gabor(c, sf, ori=45, phase=0.25, contrast=c.design.probe_contrast)
                im.set_data(img); im.set_extent([-ext / 2, ext / 2, -ext / 2, ext / 2])
                ax.set_title(f"probe {sf:.2f} cpd", fontsize=7)
            else:
                im.set_data(np.zeros((2, 2))); ax.set_title("no probe", fontsize=7, color="0.6")

        for i, name in enumerate(self.cond_names):
            cd, rs = s["conditions"].get(name), real.get(name)
            if cd is None:
                continue
            st, m = cd["stats"], cd["match"] or {}
            def pair(k, fmt):
                return f"{st[k]:{fmt}}" + (f" / {rs[k]:{fmt}}" if rs else "")
            if rs:
                off = max(abs(rs["mean_log2"] - st["mean_log2"]), abs(rs["sd_octaves"] - st["sd_octaves"]))
                check, color = (f"ok ({off:.3f} oct)", "#e6f2e6") if off <= TOL_OCT else (f"OFF by {off:.3f} oct", "#f6d5d5")
            else:
                check, color = "not run", "white"
            vals = [name, str(runs.get(name, 0)), pair("mean_cpd_geometric", ".3f"), pair("sd_octaves", ".3f"),
                    pair("skewness", "+.2f"), f"{st['entropy_bits']:.3f}", f"{st['p_above_center']:.3f}",
                    ",".join(f"{v:.2f}" for v in st["modes_cpd"]) or "none", str(m.get("matched", "-")), check]
            for j, v in enumerate(vals):
                cell = self.table[i + 1, j]
                cell.get_text().set_text(v)
                cell.set_facecolor(color if j == len(vals) - 1 else "white")
                cell.get_text().set_color("k" if runs.get(name) else "0.6")
        self.order_text.set_text(f"Block order (cb {self.ids['cb index']}): {' > '.join(s['order'])}    "
                                 f"sampling: {c.design.sampling}    seed: {s['seed']}    int/real = intended / realized")
        self.fig.canvas.draw_idle()

    # -- output -----------------------------------------------------------
    def export(self) -> Path | None:
        if not self.refresh():  # never export a stale schedule behind an error
            return None
        s = self.sched
        out = self.export_dir / f"sub-{self.ids['participant']}_ses-{self.ids['session']}_seed-{s['seed']}"
        out.mkdir(parents=True, exist_ok=True)
        self.cfg.save(out / "config.json")
        write_json(out / "schedule_summary.json", {k: v for k, v in s.items() if k != "blocks"}
                   | {"realized": {k: {kk: vv for kk, vv in v.items() if kk != "pmf"} for k, v in realized_stats(s).items()}})
        cols = [c for c in COLUMNS if c in s["blocks"][0]["trials"][0]]
        with open(out / "trial_sequence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for b in s["blocks"]:
                w.writerows(b["trials"])
        self.fig.savefig(out / "overview.png", dpi=120)
        self.msg.set_text(f"exported -> {out}")
        self.fig.canvas.draw_idle()
        return out

    def run(self) -> list[str] | None:
        """Export the current settings, then start the participant window with exactly that config."""
        out = self.export()
        if out is None:
            return None
        if importlib.util.find_spec("psychopy"):
            py = sys.executable
        else:
            py = str(Path(__file__).resolve().parents[1] / ".venv-psychopy" / "bin" / "python")
            if not Path(py).exists():
                self.msg.set_text("ERROR: PsychoPy not found; install requirements-psychopy.txt")
                self.fig.canvas.draw_idle()
                return None
        cmd = [py, "-m", "sfu.runner", "--config", str(out / "config.json"),
               "--participant", str(self.ids["participant"]), "--session", str(self.ids["session"]),
               "--cb-index", str(self.ids["cb index"])]
        if not self.cfg.monitor.calibrated:
            cmd += ["--demo", "--windowed"]  # uncalibrated monitor: test run only
        subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1])
        self.msg.set_text(f"started {self.ids['participant']} session {self.ids['session']}"
                          + ("\n(demo: monitor not calibrated)" if not self.cfg.monitor.calibrated else ""))
        self.fig.canvas.draw_idle()
        return cmd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.json")
    ap.add_argument("--export-dir", default="exports")
    a = ap.parse_args(argv)
    import matplotlib.pyplot as plt
    gui = ResearcherGUI(cfgmod.load(a.config), a.export_dir)
    plt.show()
    return gui


if __name__ == "__main__":
    main()
