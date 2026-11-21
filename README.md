# sfu — Spatial-frequency uncertainty and visual cortical tuning

Exploratory PsychoPy prototype asking: **when the upcoming spatial frequency (SF) of a stimulus is uncertain, does the visual cortex prepare for a single expected SF, or for the whole distribution of possible SFs?**

Prepared for discussion with Dr. Sam Ling's lab (Boston University). This is a **behavioral prototype**, not a validated fMRI protocol. Science: [docs/PROTOCOL.md](docs/PROTOCOL.md). Design decisions: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

---

## Research questions

**Primary**

1. **Point estimate or distribution?** With the mean expected SF held constant, does population SF tuning depend only on that mean (or mode), or on the full shape of the expected distribution?
2. **Width.** Does greater uncertainty (broad vs narrow Gaussian prior, equal mean) broaden pSFT bandwidth? This is the feature-domain analogue of the attentional-field broadening with cue width in Bloem et al. (2025).
3. **Asymmetry.** Does a skewed prior (tail toward high or low SF) produce asymmetric tuning, beyond a peak shift? Ramirez et al. (2025) modeled pSFT as symmetric log-Gaussian, so an asymmetric response would need a non-Gaussian model.
4. **Multimodality.** Does a bimodal prior (two expected SFs) produce a flattened or two-peaked population profile, or a single peak between the two?

**Mechanism**

5. **Gain vs reshaping.** Can apparent bandwidth or peak changes be explained by response gain on SF channels near the expected values (feature-similarity gain), without any change in the underlying tuning shape?
6. **Expectation vs adaptation.** Are effects driven by *knowing* the distribution (cue, learning) or by *exposure* to its stimuli (SF-selective adaptation)? The probe paradigm, which shows physically identical probes in every condition, and planned yoked-exposure blocks target this question.
7. **Expectation vs attention.** Do effects survive when the measured stimuli (probes) are task-irrelevant and in the opposite hemifield?

**Behavioral (answerable with this prototype)**

8. Is orientation discrimination accuracy or RT at a given SF better when that SF is more probable (`p_expected`), after per-SF difficulty is equated?
9. Do behavioral benefits scale with prior variance, entropy, or neither? These two measures cannot both be matched across shapes.
10. Is a histogram cue enough to establish the prior, or must it be learned from exposure over the block?

**Methodological**

11. Which matching scheme gives the cleaner comparison of shapes: mean/variance matching or mode matching? The two disagree on what "right-skewed" means (see PROTOCOL §2).
12. With realistic noise, how many probe SFs and how many trials are needed to tell Gaussian, asymmetric, and bimodal tuning apart? Synthetic model recovery in `sfu/simulation.py` suggests 13 probe levels at noise SD 0.1 are **not** enough.

Behavioral results (Q8–10) cannot, alone, answer Q1–7. Those require neural measurements such as fMRI pSFT fits with the [pSF-Toolbox](https://github.com/luisdramirez/pSF-Toolbox).

---

## What the program does

**Participant side.** On a gray screen with a central fixation cross:

1. Instructions, then practice (12 trials).
2. An optional calibration block (uniform prior), in which a 2-down/1-up staircase per SF finds the tilt giving ~71% correct. Those tilts are then **frozen**, so difficulty is equal across SFs and the staircase cannot absorb the expectation effect.
3. Six main blocks (108 trials each), one per distribution, in Williams-counterbalanced order. Each block opens with a cue (default: a histogram of the SF probabilities).
4. Each trial runs as follows: 500 ms fixation → 100 ms Gabor at 5° eccentricity → up to 1.5 s to press ← (counter-clockwise) or → (clockwise) → 0.8–1.4 s jittered ITI. There is no feedback.

The default session is about 930 trials, roughly 40 min. Pressing Escape aborts; `--resume` continues from the next unlogged trial.

**Conditions.** SF is drawn from a 9-level log2 grid from 1 to 4 cpd (center 2 cpd):

| Condition | Meaning |
|---|---|
| `narrow` | high certainty (Gaussian, SD 0.2 oct) |
| `broad` | more uncertainty, same mean (SD ≈ 0.46 oct) |
| `uniform` | every SF equally likely |
| `right_skewed` / `left_skewed` | tail toward high / low SF |
| `bimodal` | two expected SFs (1.41 and 2.83 cpd) |

In `moment_matched` mode (default), broad, skewed and bimodal share mean and variance. Each match is verified numerically (residual < 1e-6), and any condition that cannot be matched is reported as unmatched. Uniform cannot be variance-matched on a fixed grid. `mode_matched` mode instead fixes the peak at 2 cpd.

**Probe paradigm** (`design.paradigm = "probe"`). On half of the trials, a task-irrelevant Gabor appears in the mirrored hemifield, with SFs spanning 0.5–8 cpd. The probe stream is **identical across conditions**, so probe responses differ only in expectation context, not in physical stimulation.

---

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt                                # headless, GUI, analysis, tests
python3.10 -m venv .venv-psychopy && .venv-psychopy/bin/pip install -r requirements-psychopy.txt   # participant display
```

PsychoPy wheels are most reliable on Python 3.10. Everything except the participant window runs without PsychoPy.

## Run

```bash
.venv/bin/python -m pytest -q                                                    # tests
.venv/bin/python -m sfu.runner --config configs/demo.json --participant demo01 --headless   # synthetic, no display
.venv/bin/python -m sfu.analysis data/demo --out figures/demo                    # plots + summary CSVs
.venv/bin/python -m sfu.gui --config configs/demo.json                           # researcher interface
.venv/bin/python -m sfu.simulation --out figures/simulation                      # SYNTHETIC tuning models
PYTHONPATH=. .venv/bin/python scripts/diagnostics.py                             # theoretical vs sampled distributions
```

Participant display, short demo, real keypresses:

```bash
.venv-psychopy/bin/python -m sfu.runner --config configs/demo.json --participant test01 --demo --windowed
```

Add `--autorespond` to watch it run itself with synthetic responses.

**macOS keyboard caveat (unresolved).** On the development MacBook, keypresses did not reach the PsychoPy window when it was launched from an embedded terminal. Check with:

```bash
.venv-psychopy/bin/python scripts/key_test.py event   # click the window, press keys; should print "got key: ..."
```

If no keys print, launch from Terminal.app. Alternatively, grant Terminal **Input Monitoring** permission (System Settings → Privacy & Security) and use `"keyboard_backend": "ptb"`, which also gives the best RT precision.

### Behavioral pilot

1. Copy `configs/pilot_template.json` → `configs/pilot.json`. Enter the measured screen width (cm), resolution (px), viewing distance (cm) and refresh rate, then set `monitor.calibrated: true`.
2. Verify the calibration with a ruler: `.venv-psychopy/bin/python -m sfu.runner --config configs/pilot.json --check-calibration`
3. Use a fixed-refresh display (no ProMotion/VRR) and verify stimulus duration with a photodiode. The runner refuses to start if the measured refresh rate deviates >1% from the configured rate.
4. Run: `.venv-psychopy/bin/python -m sfu.runner --config configs/pilot.json --participant P01 --session 1` (counterbalancing row = trailing digits of the ID, or `--cb-index`).
5. Analyze: `.venv/bin/python -m sfu.analysis data/pilot --out figures/pilot`

---

## Changing the distributions

All distribution settings are in the `"distributions"` section of the config JSON, or can be set live in the researcher GUI.

- **Choose conditions:** `"conditions": ["narrow", "bimodal"]`
- **Reshape built-ins:** `narrow_sd`, `broad_sd`, `skew_alpha`, `bimodal_sep`, `bimodal_sd`, and `matching` (`moment_matched` / `mode_matched`)
- **Change the range:** `sf_min_cpd`, `sf_max_cpd`, `n_levels`
- **Add your own:**
  ```json
  "conditions": ["narrow", "my_lopsided"],
  "custom": {"my_lopsided": {"family": "bimodal", "params": {"loc": 1.0, "sep": 1.0, "scale": 0.2, "weight": 0.8}}}
  ```
  Families: `gaussian` (loc, scale), `uniform`, `skewnormal` (alpha, loc, scale), `bimodal` (loc, sep, scale, weight). `loc` is in log2 cpd (1 cpd = 0, 2 cpd = 1, 4 cpd = 2).

The `"design"` section controls `sampling` and the block `cue`:

- `sampling`: `fixed_count` gives exact proportions in a shuffled order; `probability` draws each trial independently.
- `cue`: `histogram` / `text` / `symbol` / `none`.

Configs are validated on load. Unknown keys, unknown conditions, and impossible values are rejected.

---

## Data

Each session writes `data/<config>/sub-<ID>_ses-<N>/`:

| File | Contents |
|---|---|
| `trials.csv` / `trials.json` | one row per trial, fsynced immediately (also serves as the resume checkpoint) |
| `session.json` | full config, software versions, measured refresh, block order, condition statistics, state history, status (`complete` / `aborted`) |
| `thresholds.json` | frozen per-SF tilts from the calibration block |

The trial columns (see `sfu/datalog.py: COLUMNS`) are grouped as follows:

- **Identifiers:** participant, session, block, trial, UID, timestamp.
- **Condition:** distribution family and parameters, matching, sampling, cue.
- **Stimulus:** SF (cpd and log2), tilt, orientation, phase, contrast, envelope, position.
- **Probabilities:** expected, realized in the block, and running.
- **Probe:** probe parameters.
- **Timing:** flip-based onset/offset, measured duration, frame count, dropped frames, max frame interval.
- **Response:** key, correct key, accuracy, RT.
- **Reproducibility:** seed, block seed, config hash, software version, backend, `synthetic`, `units_calibrated`.

Rows produced by the simulated observer (headless runs, `--autorespond`, and `example_data/`) carry `synthetic=True`.

---

## Code layout

| Module | Role |
|---|---|
| `sfu/distributions.py` | log2-SF grid; Gaussian, uniform, skew-normal, bimodal PMFs; mode- and moment-matching with numerical checks; mean, variance, entropy, skewness, kurtosis, modes |
| `sfu/calibration.py` | exact monitor geometry (atan), deg↔px, cpd→cycles/px, Nyquist guard, frame counts |
| `sfu/config.py` | dataclass config, strict validation, JSON I/O |
| `sfu/stimulus.py` | Gabor parameters → PsychoPy `GratingStim` (pixel units, own calibration); numpy preview |
| `sfu/schedule.py` | seeded schedules, Williams counterbalancing, probability vs fixed-count sampling, per-SF response balancing, identical probe streams |
| `sfu/adaptive.py` | per-SF 2-down/1-up staircases (calibration block only) |
| `sfu/state.py` | session state machine |
| `sfu/responses.py` | synthetic observer |
| `sfu/hardware.py` | scanner trigger (keyboard-emulated) and event-marker hooks (null / log / serial) |
| `sfu/datalog.py` | crash-safe CSV, JSON sidecar, versions, config hash |
| `sfu/runner.py` | session driver, PsychoPy and headless backends, CLI |
| `sfu/gui.py` | researcher interface (matplotlib widgets): distributions, stats, Gabor preview, sequence export |
| `sfu/analysis.py` | intended-vs-realized validation, accuracy (Wilson CI) and RT (bootstrap CI) by SF × condition |
| `sfu/simulation.py` | SYNTHETIC tuning scenarios (narrow, broad, gain, peak shift, asymmetric, bimodal); Gaussian, split-Gaussian, bimodal, B-spline fits; parameter and model recovery |
| `scripts/` | config generation, distribution diagnostics, keyboard check |
| `tests/test_core.py` | normalization, matched moments, reproducibility, balancing, SF bounds, complete logging, condition assignment, abort/resume, frozen thresholds |

---

## Validation status

| Check | Status |
|---|---|
| Unit tests (Python 3.12 core, 3.10 PsychoPy env) | pass |
| Headless end-to-end session + analysis | pass |
| PsychoPy window, autorespond (MacBook, 120 Hz ProMotion, windowed) | ran; 12-frame stimuli measured 96–101 ms, dropped frames detected. This display is not suitable for data collection. |
| Real keypresses on macOS | **failing on dev machine** (see keyboard caveat above) |
| Researcher GUI | tested headless only |
| Calibrated monitor, gamma, photodiode timing, keyboard latency, scanner trigger, serial markers, eye tracking | not done; requires lab hardware |

## Not implemented

Yoked exposure-matched blocks; serial trigger input; eye tracking (needed for covert attention at 5° eccentricity); practice feedback; per-SF contrast normalization; fMRI pSFT fitting (use the pSF-Toolbox; no GPL code is included here). License to be chosen with collaborators.

## Key references (read during preparation)

- Ramirez, L.D., Wang, F., & Ling, S. (2025). Attention alters population spatial frequency tuning. *J Neurosci*. doi:10.1523/JNEUROSCI.0251-25.2025
- Bloem, I.M., Bakst, L., McGuire, J.T., & Ling, S. (2025). Dynamic estimation of the attentional field from visual cortical activity. *eLife* 14:RP104222. doi:10.7554/eLife.104222
- pSF-Toolbox — Ramirez et al. (2026), *J Open Research Software* 14(1):5. doi:10.5334/jors.610

Other background literature is listed in [docs/PROTOCOL.md §7](docs/PROTOCOL.md), marked unverified where it was cited from memory.
