# sfu: Spatial-frequency uncertainty and visual cortical tuning

This is an exploratory PsychoPy prototype. It asks whether uncertainty about an upcoming spatial frequency (SF) changes population SF tuning. The idea was prepared for discussion with Dr. Sam Ling's lab (Boston University). **It is not a validated fMRI protocol.** The current implementation is a behavioral prototype that has only been tested on the developer's laptop. See [docs/PROTOCOL.md](docs/PROTOCOL.md) for the science and [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for design decisions.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt          # headless, GUI, analysis, tests
python3.10 -m venv .venv-psychopy && .venv-psychopy/bin/pip install -r requirements-psychopy.txt   # participant display
```

PsychoPy wheels are most reliable on Python 3.10. Everything except the participant window runs without PsychoPy.

## Quick demo (no display hardware needed)

```bash
.venv/bin/python -m pytest -q                                            # 21 tests
.venv/bin/python -m sfu.distributions                                    # distribution statistics
.venv/bin/python -m sfu.runner --config configs/demo.json --participant demo01 --headless
.venv/bin/python -m sfu.analysis data/demo --out figures/demo
.venv/bin/python -m sfu.simulation --out figures/simulation              # SYNTHETIC tuning + recovery
PYTHONPATH=. .venv/bin/python scripts/diagnostics.py                     # theoretical vs sampled
.venv/bin/python -m sfu.gui --config configs/demo.json                   # researcher interface
```

Headless runs use a **synthetic observer**, and every row they produce is marked `synthetic=True`. The same applies to `example_data/`.

To see the participant display with automatic synthetic responses (a timing smoke test that needs no keypresses):

```bash
.venv-psychopy/bin/python -m sfu.runner --config configs/demo.json --participant demo01 --demo --windowed --autorespond
```

On macOS, set `"keyboard_backend": "event"` in the config unless the terminal has **Input Monitoring** permission. Without that permission, the default `ptb` keyboard hangs while enumerating HID devices (this was observed during development).

## Running a behavioral pilot

1. **Calibrate.** Copy `configs/pilot_template.json`. Measure the visible screen width (cm), the resolution (px), the viewing distance (cm, chin rest) and the refresh rate. Set `monitor.calibrated: true` only after measuring all of them. Then check with a ruler:
   ```bash
   .venv-psychopy/bin/python -m sfu.runner --config configs/pilot.json --check-calibration
   ```
   On Retina/HiDPI displays, confirm that PsychoPy pixels are physical pixels (the 10 cm bar must measure 10 cm). Linearize luminance (gamma) with a photometer and record in `monitor.gamma` what was done.
2. **Display.** Run fullscreen on a fixed-refresh display, with ProMotion/VRR disabled. The runner measures the refresh rate and aborts if it is more than 1% off the configured value, unless `--demo` is given. Before collecting data, verify stimulus duration with a photodiode.
3. **Run.**
   ```bash
   .venv-psychopy/bin/python -m sfu.runner --config configs/pilot.json --participant P01 --session 1
   ```
   The counterbalancing row defaults to the trailing digits of the ID (P01 → row 1 of the Williams square). Use `--cb-index` to set it explicitly. Press Escape to abort. The data written up to that point is kept, and the run continues with:
   ```bash
   .venv-psychopy/bin/python -m sfu.runner --config configs/pilot.json --participant P01 --session 1 --resume
   ```
   Resume is refused if the config or the counterbalancing index has changed.
4. **Analyze.**
   ```bash
   .venv/bin/python -m sfu.analysis data/pilot --out figures/pilot
   ```

Default session: 12 practice trials, then a 270-trial calibration block (staircases), then 6 × 108 main trials. That is about 930 trials, roughly 40 min. Each block opens with a histogram cue showing the SF distribution.

## Layout

| Module | Role |
|---|---|
| `sfu/distributions.py` | log2-SF grid; Gaussian, uniform, skew-normal and bimodal PMFs; mode- and moment-matching with numerical checks; mean, variance, entropy, skewness, kurtosis, modes |
| `sfu/calibration.py` | monitor geometry (exact atan), deg↔px, cpd→cycles/px, Nyquist guard, frame counts |
| `sfu/config.py` | dataclass config, strict validation (unknown keys rejected), JSON I/O |
| `sfu/stimulus.py` | Gabor parameters → PsychoPy `GratingStim` (pixel units, own calibration); numpy preview |
| `sfu/schedule.py` | seeds, Williams counterbalancing, probability vs fixed-count sampling, per-SF response balancing, identical probe streams |
| `sfu/adaptive.py` | per-SF 2-down/1-up staircases (calibration block only, then frozen) |
| `sfu/state.py` | session state machine |
| `sfu/responses.py` | synthetic observer (headless / autorespond); keyboard handling is in the PsychoPy backend |
| `sfu/hardware.py` | scanner trigger (keyboard-emulated) and event-marker hooks (null / log / serial) |
| `sfu/datalog.py` | fsynced per-trial CSV (the checkpoint), JSON sidecar, versions, config hash |
| `sfu/runner.py` | session driver, PsychoPy and headless backends, CLI |
| `sfu/gui.py` | researcher interface |
| `sfu/analysis.py` | realized-vs-intended validation, accuracy (Wilson CI) and RT (bootstrap CI) by SF × condition |
| `sfu/simulation.py` | SYNTHETIC tuning scenarios, Gaussian, split-Gaussian, bimodal and B-spline fits, parameter and model recovery |

## Data

Each session writes `data/<out>/sub-<id>_ses-<n>/`:

- `trials.csv` / `trials.json`: one row per trial. Columns are listed in `sfu/datalog.py: COLUMNS`. They cover IDs, condition and distribution parameters, SF (cpd and log2), orientation, tilt, phase, contrast, position, expected and realized probabilities, flip-based onset and offset, measured duration, response, accuracy, RT, dropped frames, seeds, config hash, version, backend, a synthetic flag, and a units-calibrated flag.
- `session.json`: the full config, software versions, measured refresh, block order, cue symbols, condition statistics, state-machine history, and status (`complete` / `aborted`).
- `thresholds.json`: the frozen per-SF tilts from the calibration block.

## Validation status

| Check | Status |
|---|---|
| Unit tests (21): normalization, matched moments, reproducibility, balancing, SF bounds, complete logging, condition assignment, abort/resume, frozen thresholds | **pass** on Python 3.12 (core) and 3.10 (PsychoPy env) |
| Headless end-to-end session + analysis | **pass** |
| PsychoPy windowed autorespond run (38 trials, probe paradigm) | **ran** on a MacBook (120 Hz ProMotion, windowed). Duration 96–101 ms for 12 frames; 15 dropped frames detected over 38 trials. Dropped-frame detection works; this display is **not** suitable for data collection. |
| Researcher GUI | **tested headless** (Agg backend: widget callbacks, export). Interactive window not exercised in this environment. |
| Calibrated monitor, gamma, photodiode timing, real keyboard latency, scanner trigger, serial markers, eye tracking | **not done here: requires lab hardware** |

## Not implemented (deliberately)

The following are named in [PROTOCOL.md §5/§8](docs/PROTOCOL.md):

- yoked exposure-matched blocks
- serial trigger *input*
- eye tracking
- per-SF contrast normalization
- fMRI pSFT fitting (use the [pSF-Toolbox](https://github.com/luisdramirez/pSF-Toolbox))

The license is still to be chosen with collaborators. No pSF-Toolbox (GPL-3.0) code is included.
