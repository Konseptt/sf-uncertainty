"""Session driver + backends + CLI.

    python -m sfu.runner --config configs/pilot.json --participant P01 --session 1 --cb-index 0
    python -m sfu.runner --config configs/demo.json --participant demo --headless      # no display
    python -m sfu.runner --config configs/demo.json --participant demo --demo --windowed --autorespond
    python -m sfu.runner --config configs/pilot.json --check-calibration
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as cfgmod
from .adaptive import Staircase
from .config import ExperimentConfig
from .datalog import TrialLogger, completed_uids, config_hash, csv_to_json, software_versions, write_json
from .hardware import make_markers, wait_keyboard_trigger
from .responses import SimulatedObserver
from .schedule import build_session
from .state import Aborted, State, StateMachine
from .stimulus import gabor_kwargs

log = logging.getLogger("sfu")

INSTRUCTIONS = ("Keep your eyes on the central cross.\n\nA striped patch will flash briefly.\n"
                "Press LEFT if it is tilted counter-clockwise, RIGHT if clockwise.\n\n"
                "Respond as accurately and quickly as you can.\n\nPress any key to start.")
CUE_TEXT = {
    "narrow": "Stripes in this block will almost always have the same fineness.",
    "broad": "Stripe fineness in this block will vary moderately around a typical value.",
    "uniform": "Every stripe fineness is equally likely in this block.",
    "right_skewed": "Stripes in this block are usually medium, sometimes much finer.",
    "left_skewed": "Stripes in this block are usually medium, sometimes much coarser.",
    "bimodal": "Stripes in this block are either fairly coarse or fairly fine.",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class HeadlessBackend:
    """No window, no PsychoPy. Responses from a SYNTHETIC observer; timings are
    nominal frame counts. For testing logic only."""
    name = "headless"

    def __init__(self, cfg: ExperimentConfig, observer: SimulatedObserver):
        self.cfg, self.obs, self.t = cfg, observer, 0.0

    def instructions(self, text): pass
    def block_intro(self, block, sched): pass
    def block_end(self, block): pass
    def wait_trigger(self): return self.t
    def close(self): pass

    def run_trial(self, trial: dict, tilt: float) -> dict:
        m, s = self.cfg.monitor, self.cfg.stimulus
        fr = 1 / m.refresh_hz
        n_stim = m.frames(s.duration_ms)
        self.t += m.frames(s.fixation_ms) * fr
        onset = self.t
        key, rt = self.obs.respond(trial, tilt, (self.cfg.key_left, self.cfg.key_right))
        if rt > s.response_window_ms / 1000:
            key, rt = "", math.nan
        self.t += max(n_stim * fr, rt if rt == rt else s.response_window_ms / 1000) + m.frames(trial["iti_ms"]) * fr
        return dict(stim_frames=n_stim, onset_s=onset, offset_s=onset + n_stim * fr,
                    duration_measured_ms=n_stim * fr * 1000, response=key, rt_s=rt,
                    dropped_frames=0, frame_interval_max_ms=fr * 1000, backend=self.name, synthetic=True)


class PsychoPyBackend:
    name = "psychopy"

    def __init__(self, cfg: ExperimentConfig, windowed: bool = False, autorespond: SimulatedObserver | None = None,
                 demo: bool = False):
        from psychopy import core, event, visual
        from psychopy.hardware import keyboard
        self.cfg, self.auto = cfg, autorespond
        m, s = cfg.monitor, cfg.stimulus
        size = (m.width_px, m.height_px) if not windowed else (min(1280, m.width_px), min(800, m.height_px))
        self.win = visual.Window(size=size, fullscr=not windowed, units="pix", color=[s.background_rgb] * 3,
                                 colorSpace="rgb", allowGUI=False, waitBlanking=True)
        self.win.mouseVisible = False
        try:  # macOS: window launched from a terminal does not get keyboard focus by itself
            self.win.winHandle.activate()
        except AttributeError:
            pass
        hz = self.win.getActualFrameRate(nIdentical=20, nMaxFrames=240, nWarmUpFrames=20)
        self.measured_hz = hz
        if hz is None or abs(hz - m.refresh_hz) / m.refresh_hz > cfg.max_frame_rate_error:
            msg = f"measured refresh {hz} Hz vs configured {m.refresh_hz} Hz"
            if not demo or hz is None:
                self.win.close()
                raise RuntimeError(msg + " (use --demo to override for testing)")
            log.warning(msg + "; DEMO: using measured %d Hz for frame counts", round(hz))
            cfg.monitor = m = replace(m, refresh_hz=float(round(hz)))
        self.fr = 1 / m.refresh_hz
        self.win.recordFrameIntervals = True
        fix = m.deg2px(s.fixation_size_deg)
        self.fix = visual.ShapeStim(self.win, vertices="cross", size=fix, lineWidth=0, fillColor=-1, units="pix")
        init = dict(sf_cpd=2.0, ori=0.0, phase=0.0, contrast=s.contrast, pos_deg=(0, 0))
        self.target = visual.GratingStim(self.win, **gabor_kwargs(cfg, **init))
        self.probe = visual.GratingStim(self.win, **gabor_kwargs(cfg, **init))
        self.text = visual.TextStim(self.win, height=m.deg2px(0.6), color=-1, wrapWidth=m.deg2px(20), units="pix")
        # psychopy 2026.2.5 Keyboard(backend="event") is broken: no key-up events (so the default
        # waitRelease=True never returns a key), names come back as [name, t] lists (keyList never
        # matches) and rt is the poll time. Use psychopy.event directly for that backend.
        # ponytail: event-backend RT is stamped when pyglet dispatches (once per frame, ~8-17 ms
        # resolution); use "ptb" for real RT data.
        self.event = event if cfg.keyboard_backend == "event" else None
        self.kb = None if self.event else keyboard.Keyboard(backend=cfg.keyboard_backend)
        self.rt_clock = core.Clock() if self.event else self.kb.clock
        self.markers = make_markers(cfg.design.markers, cfg.design.serial_port)

    def _set(self, stim, sf, ori, phase, contrast, pos_deg):
        k = gabor_kwargs(self.cfg, sf, ori, phase, contrast, pos_deg)
        stim.sf, stim.size, stim.ori, stim.phase, stim.contrast, stim.pos = k["sf"], k["size"], ori, phase, contrast, k["pos"]

    def _keys(self, keyList=None) -> list[tuple[str, float]]:
        """Key presses since last poll as (name, rt from last rt_clock reset); reported on press."""
        if self.event:
            return [(n, float(t)) for n, t in self.event.getKeys(keyList=keyList, timeStamped=self.rt_clock)]
        return [(k.name, float(k.rt)) for k in self.kb.getKeys(keyList=keyList, waitRelease=False)]

    def _clear_keys(self):
        if self.event:
            self.event.clearEvents("keyboard")
        else:
            self.kb.clearEvents()

    def _wait_key(self, seconds_if_auto=0.3):
        if self.auto:
            time.sleep(seconds_if_auto)
            return
        self._clear_keys()
        while True:
            ks = self._keys()
            if ks:
                if ks[0][0] == self.cfg.key_quit:
                    raise Aborted()
                return
            time.sleep(0.005)

    def _check_quit(self):
        if self._keys(keyList=[self.cfg.key_quit]):
            raise Aborted()

    def instructions(self, text):
        self.text.text = text
        self.text.draw(); self.win.flip()
        self._wait_key()

    def block_intro(self, block, sched):
        from psychopy import visual
        cue, cond = self.cfg.design.cue, block["condition"]
        m = self.cfg.monitor
        if block["block_type"] != "main":
            msg = {"practice": "Practice block.", "calibration": "Calibration block."}[block["block_type"]]
            self.text.text, self.text.pos = msg + "\n\nPress any key to start.", (0, 0)
            self.text.draw()
        elif cue == "histogram":
            p = np.asarray(block["pmf"]); w = m.deg2px(1.0); h = m.deg2px(6.0)
            for i, pi in enumerate(p):
                x = (i - (len(p) - 1) / 2) * w * 1.1
                visual.Rect(self.win, width=w, height=max(1, h * pi / p.max()), pos=(x, h * pi / p.max() / 2 - h / 2),
                            fillColor=-0.6, lineWidth=0, units="pix").draw()
            self.text.text, self.text.pos = "coarse stripes                                      fine stripes\n\nPress any key to start.", (0, -h / 2 - m.deg2px(1.5))
            self.text.draw()
        elif cue == "text":
            self.text.text, self.text.pos = CUE_TEXT.get(cond, "") + "\n\nPress any key to start.", (0, 0)
            self.text.draw()
        elif cue == "symbol":
            self.text.text, self.text.pos = f"{block['symbol']}\n\nPress any key to start.", (0, 0)
            self.text.draw()
        else:
            self.text.text, self.text.pos = "Press any key to start.", (0, 0)
            self.text.draw()
        self.win.flip()
        self._wait_key(0.5)
        self.text.pos = (0, 0)
        self.markers.send("block_start")

    def block_end(self, block):
        self.markers.send("block_end")
        self.text.text = "Block finished. Take a short break.\n\nPress any key to continue."
        self.text.draw(); self.win.flip()
        self._wait_key()

    def wait_trigger(self):
        self.text.text = "Waiting for scanner..."
        self.text.draw(); self.win.flip()
        d = self.cfg.design
        if d.trigger == "keyboard":
            return wait_keyboard_trigger(self.win, d.trigger_key, self.cfg.key_quit)
        raise NotImplementedError("serial trigger input: implement for your scanner interface")

    def run_trial(self, trial: dict, tilt: float) -> dict:
        m, s = self.cfg.monitor, self.cfg.stimulus
        n_fix, n_stim = m.frames(s.fixation_ms), m.frames(s.duration_ms)
        n_resp = max(m.frames(s.response_window_ms), n_stim + 1)
        ori = s.base_orientation_deg + trial["tilt_dir"] * tilt
        self._set(self.target, trial["sf_cpd"], ori, trial["phase"], trial["contrast"], (trial["pos_x_deg"], trial["pos_y_deg"]))
        if trial["probe_present"]:
            self._set(self.probe, trial["probe_sf_cpd"], trial["probe_orientation_deg"], trial["probe_phase"],
                      trial["contrast"], (trial["probe_pos_x_deg"], trial["probe_pos_y_deg"]))
        plan = self.auto.respond(trial, tilt, (self.cfg.key_left, self.cfg.key_right)) if self.auto else None
        self.win.frameIntervals = []
        self.markers.send("fixation")
        for _ in range(n_fix):
            self.fix.draw(); self.win.flip()
        self._check_quit()
        onset = offset = None
        resp = None
        for f in range(n_resp):
            if f < n_stim:
                self.target.draw()
                if trial["probe_present"]:
                    self.probe.draw()
            self.fix.draw()
            if f == 0:
                self.win.callOnFlip(self._clear_keys)
                self.win.callOnFlip(self.rt_clock.reset)
                self.win.callOnFlip(self.markers.send, "stim_on")
            if f == n_stim:
                self.win.callOnFlip(self.markers.send, "stim_off")
            t = self.win.flip()
            if f == 0:
                onset = t
            if f == n_stim:
                offset = t
            for name, rt in self._keys(keyList=[self.cfg.key_left, self.cfg.key_right, self.cfg.key_quit]):
                if name == self.cfg.key_quit:
                    raise Aborted()
                if resp is None and plan is None:
                    resp = (name, rt)
                    self.markers.send("response")
            if plan and resp is None and plan[1] <= (f + 1) * self.fr:
                resp = plan
            if resp and f >= n_stim:
                break
        for _ in range(m.frames(trial["iti_ms"])):
            self.fix.draw(); self.win.flip()
        self._check_quit()
        iv = np.asarray(self.win.frameIntervals[1:] if self.win.frameIntervals else [0.0])
        key, rt = resp if resp else ("", math.nan)
        return dict(stim_frames=n_stim, onset_s=onset, offset_s=offset, duration_measured_ms=(offset - onset) * 1000,
                    response=key, rt_s=rt, dropped_frames=int((iv > 1.5 * self.fr).sum()),
                    frame_interval_max_ms=float(iv.max() * 1000), backend=self.name, synthetic=plan is not None)

    def close(self):
        self.markers.close()
        self.win.close()


def run_session(cfg: ExperimentConfig, participant: str, session: int, cb_index: int, backend,
                out_dir: str | Path, resume: bool = False, demo: bool = False,
                abort_after: int | None = None) -> Path:
    """Run (or resume) a session. Returns the output directory."""
    if not cfg.monitor.calibrated and not demo and backend.name != "headless":
        raise RuntimeError("monitor.calibrated is false: SF in cpd would be unverified. "
                           "Measure the display and set calibrated=true, or pass --demo.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", participant):
        raise ValueError("participant id: letters, digits, _ and - only")
    sched = build_session(cfg, participant, session, cb_index)
    out = Path(out_dir) / f"sub-{participant}_ses-{session}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path, side_path, thr_path = out / "trials.csv", out / "session.json", out / "thresholds.json"
    chash = config_hash(cfg.to_dict())
    if csv_path.exists() and not resume:
        raise FileExistsError(f"{csv_path} exists; use --resume or a new session number")
    if resume and side_path.exists():
        prev = json.loads(side_path.read_text())
        if prev["config_hash"] != chash or prev["cb_index"] != cb_index:
            raise RuntimeError("resume refused: config or counterbalance index differs from original run")
    done = completed_uids(csv_path) if resume else set()

    d, s = cfg.design, cfg.stimulus
    tilts = {lv: s.tilt_deg for lv in range(cfg.distributions.n_levels)}
    if thr_path.exists():
        tilts.update({int(k): v for k, v in json.loads(thr_path.read_text())["tilt_by_level"].items()})
    stairs = {lv: Staircase(tilt=d.staircase_start_tilt, step=d.staircase_step, min_step=d.staircase_min_step)
              for lv in tilts}
    if resume and csv_path.exists():  # replay logged calibration trials into staircases
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                if r["block_type"] == "calibration" and r["accuracy"] not in ("", "nan"):
                    stairs[int(r["level_index"])].update(float(r["accuracy"]) == 1.0)

    sm = StateMachine()
    side = {"participant": participant, "session": session, "cb_index": cb_index, "seed": sched["seed"],
            "config_hash": chash, "config": cfg.to_dict(), "software": software_versions(),
            "backend": backend.name, "measured_refresh_hz": getattr(backend, "measured_hz", None),
            "block_order": sched["order"], "grid_log2": sched["grid_log2"], "cue_symbols": sched["symbols"], "conditions": sched["conditions"],
            "status": "running", "started": _now(), "resumed": resume}
    write_json(side_path, side)
    logger = TrialLogger(csv_path)
    common = dict(config_hash=chash, software_version=side["software"]["sfu"], units_calibrated=cfg.monitor.calibrated)
    n_new = 0
    try:
        sm.to(State.INSTRUCTIONS)
        backend.instructions(INSTRUCTIONS)
        for block in sched["blocks"]:
            todo = [t for t in block["trials"] if t["trial_uid"] not in done]
            if not todo:
                continue
            sm.to(State.BLOCK_INTRO)
            backend.block_intro(block, sched)
            if d.trigger != "none" and block["block_type"] == "main":
                sm.to(State.WAIT_TRIGGER)
                side.setdefault("triggers", []).append({"block": block["block_index"], "t": backend.wait_trigger()})
            sm.to(State.TRIALS)
            for t in todo:
                lv, bt = t["level_index"], block["block_type"]
                tilt = d.staircase_start_tilt if bt == "practice" else stairs[lv].tilt if bt == "calibration" else tilts[lv]
                res = backend.run_trial(t, tilt)
                acc = math.nan if not res["response"] else float(res["response"] == t["correct_key"])
                logger.write({**t, **res, **common, "tilt_deg": tilt,
                              "orientation_deg": s.base_orientation_deg + t["tilt_dir"] * tilt,
                              "accuracy": acc, "timestamp": _now()})
                if bt == "calibration" and acc == acc:
                    stairs[lv].update(acc == 1.0)
                n_new += 1
                if abort_after is not None and n_new >= abort_after:
                    raise Aborted()
            if block["block_type"] == "calibration":
                tilts = {lv: sc.threshold() for lv, sc in stairs.items()}
                write_json(thr_path, {"tilt_by_level": tilts, "reversals": {lv: sc.reversals for lv, sc in stairs.items()},
                                      "note": "2-down/1-up, ~70.7% correct; frozen for main blocks"})
            sm.to(State.BLOCK_END)
            backend.block_end(block)
        sm.to(State.DONE)
        side["status"] = "complete"
    except Aborted:
        sm.to(State.ABORTED)
        side["status"] = "aborted"
        log.warning("aborted after %d new trials; resume with --resume", n_new)
    finally:
        logger.close()
        side.update(finished=_now(), state_history=sm.history, trials_logged_this_run=n_new)
        write_json(side_path, side)
        csv_to_json(csv_path, out / "trials.json")
        backend.close()
    return out


def calibration_check(cfg: ExperimentConfig, windowed: bool) -> None:
    """Draw a 10 cm bar and a 5 deg bar; measure them with a ruler."""
    from psychopy import visual, event
    m = cfg.monitor
    win = visual.Window(size=(m.width_px, m.height_px), fullscr=not windowed, units="pix", color=[cfg.stimulus.background_rgb] * 3)
    visual.Rect(win, width=10 / m.cm_per_px, height=6, pos=(0, 60), fillColor=-1, lineWidth=0).draw()
    visual.Rect(win, width=m.deg2px(5), height=6, pos=(0, -60), fillColor=-1, lineWidth=0).draw()
    expected_5deg_cm = 2 * m.distance_cm * math.tan(math.radians(2.5))
    visual.TextStim(win, text=f"Top bar must measure 10.0 cm.\nBottom bar (5 deg) must measure {expected_5deg_cm:.2f} cm "
                    f"at {m.distance_cm} cm.\n{m.px_per_deg:.2f} px/deg. Any key to exit.", pos=(0, -200), height=22, color=-1).draw()
    win.flip()
    event.waitKeys()
    win.close()


def _cb_from_id(pid: str) -> int:
    m = re.search(r"(\d+)$", pid)
    if not m:
        raise SystemExit("cannot infer --cb-index from participant id; pass --cb-index")
    return int(m.group(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--participant", default="demo01")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--cb-index", type=int, default=None, help="counterbalancing row (default: trailing digits of id)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--headless", action="store_true", help="no window; synthetic observer")
    ap.add_argument("--demo", action="store_true", help="allow uncalibrated monitor / refresh mismatch")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--autorespond", action="store_true", help="PsychoPy window with synthetic responses (timing smoke test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--abort-after", type=int, default=None, help="testing: simulate abort after N trials")
    ap.add_argument("--check-calibration", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = cfgmod.load(a.config)
    if a.check_calibration:
        return calibration_check(cfg, a.windowed)
    cb = a.cb_index if a.cb_index is not None else _cb_from_id(a.participant)
    seed = np.random.SeedSequence([cfg.study_seed, cb, 99])
    obs = SimulatedObserver(np.random.default_rng(seed))
    backend = HeadlessBackend(cfg, obs) if a.headless else PsychoPyBackend(cfg, a.windowed, obs if a.autorespond else None, a.demo)
    out = run_session(cfg, a.participant, a.session, cb, backend, a.out or cfg.output_dir, a.resume, a.demo, a.abort_after)
    print(out)


if __name__ == "__main__":
    main()
