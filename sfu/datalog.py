"""Crash-safe trial logging: one CSV row per trial (flushed + fsynced), plus a
JSON sidecar with config, schedule summary, seeds, versions and state history.
The CSV is the checkpoint used for abort recovery."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from . import __version__

COLUMNS = [
    "participant", "session", "cb_index", "block_index", "block_type", "block_rep", "trial_index", "trial_uid",
    "condition", "matching", "sampling", "dist_family", "dist_params", "cue",
    "level_index", "sf_cpd", "sf_log2", "tilt_dir", "tilt_deg", "orientation_deg", "phase", "contrast",
    "envelope_sd_deg", "pos_x_deg", "pos_y_deg", "ecc_deg",
    "p_expected", "p_observed_block", "p_observed_running",
    "probe_present", "probe_sf_cpd", "probe_sf_log2", "probe_orientation_deg", "probe_phase",
    "probe_pos_x_deg", "probe_pos_y_deg",
    "iti_ms", "stim_frames", "onset_s", "offset_s", "duration_measured_ms",
    "response", "correct_key", "accuracy", "rt_s", "dropped_frames", "stim_dropped_frames", "frame_interval_max_ms",
    "seed", "block_seed", "config_hash", "software_version", "backend", "synthetic", "units_calibrated",
    "timestamp",
]


def config_hash(cfg_dict: dict) -> str:
    return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True, default=str).encode()).hexdigest()[:16]


def software_versions() -> dict:
    v = {"sfu": __version__, "python": platform.python_version(), "numpy": np.__version__,
         "platform": platform.platform()}
    try:
        import psychopy
        v["psychopy"] = psychopy.__version__
    except ImportError:
        v["psychopy"] = "not installed"
    try:
        v["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=Path(__file__).parent, timeout=5).stdout.strip() or "none"
    except (OSError, subprocess.SubprocessError):
        v["git_commit"] = "none"
    return v


class TrialLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        new = not self.path.exists() or self.path.stat().st_size == 0
        if not new:
            with open(self.path, newline="") as f:
                if next(csv.reader(f), []) != COLUMNS:
                    raise RuntimeError(f"{self.path} was written with different columns; start a new session")
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=COLUMNS, extrasaction="raise")
        if new:
            self.w.writeheader()

    def write(self, row: dict) -> None:
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            raise KeyError(f"trial row missing columns: {missing}")
        self.w.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


def completed_uids(path: Path) -> set[int]:
    if not Path(path).exists():
        return set()
    with open(path, newline="") as f:
        return {int(r["trial_uid"]) for r in csv.DictReader(f)}


def write_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)  # atomic


def csv_to_json(csv_path: Path, json_path: Path) -> None:
    with open(csv_path, newline="") as f:
        write_json(json_path, {"trials": list(csv.DictReader(f))})
