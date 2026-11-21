"""Researcher interface: set distributions, monitor and design values, inspect
statistics, preview Gabors at the monitor's pixels per degree, and generate and
export trial sequences.

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
from .config import ExperimentConfig
from .datalog import COLUMNS, write_json
from .schedule import build_session
from .stimulus import render_gabor

FIELDS = [  # (label, getter path, type)
    ("participant", None, str), ("session", None, int), ("cb index", None, int),
    ("trials/block", "design.trials_per_block", int), ("study seed", "study_seed", int),
    ("distance cm", "monitor.distance_cm", float), ("width cm", "monitor.width_cm", float),
    ("width px", "monitor.width_px", int), ("refresh Hz", "monitor.refresh_hz", float),
    ("stim ms", "stimulus.duration_ms", float), ("contrast", "stimulus.contrast", float),
    ("ecc deg", "stimulus.eccentricity_deg", float),
]
SLIDERS = [("narrow_sd", 0.05, 0.6), ("broad_sd", 0.2, 1.5), ("skew_alpha", 0.0, 15.0),
           ("bimodal_sd", 0.05, 0.4), ("bimodal_sep", 0.4, 1.8)]


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


class ResearcherGUI:
    def __init__(self, cfg: ExperimentConfig, export_dir: str = "exports"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
        self.cfg, self.export_dir = copy.deepcopy(cfg), Path(export_dir)
        self.ids = {"participant": "P01", "session": 1, "cb index": 0}
        self.sched = None
        self.fig = fig = plt.figure(figsize=(16, 9.5))
        fig.canvas.manager and fig.canvas.manager.set_window_title("SF uncertainty - researcher interface")
        x0 = 0.015
        fig.text(x0, 0.965, "Distributions", weight="bold")
        self.r_match = RadioButtons(fig.add_axes([x0, 0.875, 0.12, 0.08]), ["mode_matched", "moment_matched"],
                                    active=["mode_matched", "moment_matched"].index(cfg.distributions.matching))
        self.r_samp = RadioButtons(fig.add_axes([x0 + 0.125, 0.875, 0.10, 0.08]), ["fixed_count", "probability"],
                                   active=["fixed_count", "probability"].index(cfg.design.sampling))
        self.sliders = {}
        for i, (k, lo, hi) in enumerate(SLIDERS):
            ax = fig.add_axes([x0 + 0.07, 0.835 - i * 0.032, 0.14, 0.022])
            self.sliders[k] = Slider(ax, k, lo, hi, valinit=getattr(cfg.distributions, k))
            self.sliders[k].on_changed(lambda _v: self.refresh())
        fig.text(x0, 0.655, "Session / monitor / timing", weight="bold")
        self.boxes = {}
        for i, (lab, path, _t) in enumerate(FIELDS):
            ax = fig.add_axes([x0 + 0.07 + (i % 2) * 0.115, 0.615 - (i // 2) * 0.034, 0.045, 0.026])
            val = self.ids[lab] if path is None else _get(cfg, path)
            self.boxes[lab] = TextBox(ax, lab + " ", initial=str(val))
            self.boxes[lab].on_submit(lambda _t: self.refresh())
        self.b_gen = Button(fig.add_axes([x0, 0.36, 0.11, 0.04]), "Generate sequence")
        self.b_exp = Button(fig.add_axes([x0 + 0.12, 0.36, 0.11, 0.04]), "Export")
        self.b_gen.on_clicked(lambda _e: self.generate())
        self.b_exp.on_clicked(lambda _e: self.export())
        self.b_run = Button(fig.add_axes([x0, 0.31, 0.23, 0.04]), "Run session with these settings")
        self.b_run.on_clicked(lambda _e: self.run())
        self.r_match.on_clicked(lambda _l: self.refresh())
        self.r_samp.on_clicked(lambda _l: self.refresh())
        self.msg = fig.text(x0, 0.27, "", fontsize=8, va="top", family="monospace")
        self.ax_d = [fig.add_axes([0.29 + (i % 3) * 0.155, 0.62 - (i // 3) * 0.27, 0.135, 0.2]) for i in range(6)]
        self.ax_g = [fig.add_axes([0.765 + i * 0.075, 0.74, 0.07, 0.17]) for i in range(3)]
        self.ax_t = fig.add_axes([0.29, 0.02, 0.69, 0.27]); self.ax_t.axis("off")
        self.refresh()

    # -- state ----------------------------------------------------------
    def _read(self) -> ExperimentConfig:
        c = self.cfg
        dist = replace(c.distributions, matching=self.r_match.value_selected,
                       **{k: float(s.val) for k, s in self.sliders.items()})
        c = replace(c, distributions=dist, design=replace(c.design, sampling=self.r_samp.value_selected))
        for lab, path, typ in FIELDS:
            v = typ(self.boxes[lab].text)
            if path is None:
                self.ids[lab] = v
            else:
                c = _set(c, path, v)
        return c.validate()

    def refresh(self) -> bool:
        """Rebuild schedule from the widgets. False (and an ERROR line) if any input is invalid."""
        try:
            self.cfg = self._read()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", str(self.ids["participant"])):
                raise ValueError("participant id: letters, digits, _ and - only")
            self.sched = build_session(self.cfg, str(self.ids["participant"]), int(self.ids["session"]), int(self.ids["cb index"]))
            self.msg.set_text("")
        except (ValueError, KeyError) as e:
            self.msg.set_text(f"ERROR: {e}")
            self.fig.canvas.draw_idle()
            return False
        self._draw()
        return True

    def generate(self):
        if self.refresh():
            n = sum(len(b["trials"]) for b in self.sched["blocks"])
            self.msg.set_text(f"seed {self.sched['seed']}\n{n} trials\norder: {' > '.join(self.sched['order'])}")
            self.fig.canvas.draw_idle()

    def _draw(self):
        s, c = self.sched, self.cfg
        g = np.array(s["grid_log2"])
        realized = {b["condition"]: np.bincount([t["level_index"] for t in b["trials"]], minlength=len(g))
                    for b in s["blocks"] if b["block_type"] == "main"}
        for ax, name in zip(self.ax_d, list(s["conditions"])[:6]):
            ax.clear()
            cond = s["conditions"][name]
            p = np.array(cond["pmf"])
            ax.bar(g, p, width=0.2, color="0.75", label="intended")
            if name in realized:
                r = realized[name] / realized[name].sum()
                ax.plot(g, r, "o", ms=4, color="k", label="realized")
            m = cond["match"]
            flag = "" if not m or "matched" not in m else ("  [matched]" if m["matched"] else "  [not matched]")
            ax.set_title(name + flag, fontsize=9)
            ax.set_xticks(g[::2], [f"{2**v:.2g}" for v in g[::2]], fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("SF (cpd, log axis)", fontsize=7)
            ax.set_ylim(0, max(0.5, p.max() * 1.15))
        self.ax_d[0].legend(fontsize=7, frameon=False)
        for ax, sf in zip(self.ax_g, (2 ** g[0], 2 ** g[len(g) // 2], 2 ** g[-1])):
            ax.clear()
            img, ext = render_gabor(c, sf, ori=0, phase=0.25,
                                    field_deg=6 * (c.stimulus.envelope_sd_deg if c.stimulus.envelope == "fixed_deg" else c.stimulus.envelope_sd_cycles / c.distributions.sf_min_cpd))
            ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest", extent=[-ext / 2, ext / 2, -ext / 2, ext / 2])
            ax.set_title(f"{sf:.2f} cpd", fontsize=8); ax.tick_params(labelsize=6)
        self.ax_g[1].set_xlabel(f"deg  ({c.monitor.px_per_deg:.1f} px/deg, {'CALIBRATED' if c.monitor.calibrated else 'UNCALIBRATED'})", fontsize=7)
        self.ax_t.clear(); self.ax_t.axis("off")
        cols = ["condition", "geo-mean cpd", "SD oct", "entropy bits", "skew", "ex. kurt", "P(>2cpd)", "modes cpd", "matched"]
        rows = []
        for n, cd in s["conditions"].items():
            st, m = cd["stats"], cd["match"] or {}
            rows.append([n, f"{st['mean_cpd_geometric']:.3f}", f"{st['sd_octaves']:.3f}", f"{st['entropy_bits']:.3f}",
                         f"{st['skewness']:+.3f}", f"{st['excess_kurtosis']:+.3f}", f"{st['p_above_center']:.3f}",
                         ",".join(f"{v:.2f}" for v in st["modes_cpd"]) or "none", str(m.get("matched", "-"))])
        tb = self.ax_t.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.25)
        self.ax_t.text(0, -0.02, f"Block order (cb {self.ids['cb index']}): {' > '.join(s['order'])}    "
                       f"sampling: {c.design.sampling}    seed: {s['seed']}", fontsize=8, transform=self.ax_t.transAxes)
        self.fig.canvas.draw_idle()

    def export(self) -> Path | None:
        if not self.refresh():  # never export a stale schedule behind an error
            return None
        s = self.sched
        out = self.export_dir / f"sub-{self.ids['participant']}_ses-{self.ids['session']}_seed-{s['seed']}"
        out.mkdir(parents=True, exist_ok=True)
        self.cfg.save(out / "config.json")
        write_json(out / "schedule_summary.json", {k: v for k, v in s.items() if k != "blocks"})
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
