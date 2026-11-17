# Scientific and technical implementation plan

Status: exploratory prototype for discussion with the Ling lab. This is not a finalized or pre-registered protocol.

## 1. What the referenced work establishes (read for this plan)

| Source | What was checked | Relevance here |
|---|---|---|
| Ramirez, Wang & Ling (2025), *J Neurosci*, doi:10.1523/JNEUROSCI.0251-25.2025 | Full text (PDF from lab site) | They used voxel-wise population SF tuning (pSFT). Each voxel's response was modeled as a **log-Gaussian** in SF (Eq. 1: peak *m*, bandwidth *s*), convolved with an HRF. SF mapping used band-limited noise at log-spaced center SFs from 0.1 to 12 cpd in a **task-irrelevant hemifield**. Participants covertly attended low-SF (0.5 cpd) or high-SF (2 cpd) letter streams. Results: **attractive peak shifts** toward the attended SF, sharper bandwidth, and peak change correlated negatively with bandwidth change. Voxels showed heterogeneous "modulation strategies". |
| Bloem, Bakst, McGuire & Ling (2025), *eLife* 14:RP104222, doi:10.7554/eLife.104222 | Full text (PDF from lab site) | The width of the cued region was varied (spatial uncertainty), and the spatial attentional modulation was fit with a **generalized Gaussian** (location μ, scale σ, shape β). FWHM broadened with cue width (V2, V3), and gain decreased with width in V3. This is the spatial analogue of the present question in the feature domain, and it shows that a non-Gaussian shape parameter can be fit. |
| pSF-Toolbox (github.com/luisdramirez/pSF-Toolbox), Ramirez et al. 2026, *JORS* 14(1):5, doi:10.5334/jors.610 | README | MATLAB, GPL-3.0. Log-Gaussian pSFT with peak, bandwidth, amplitude and baseline, plus HRF convolution and fmincon fitting. It includes simulation and validation scripts. **We do not reimplement fMRI pSFT fitting.** Our simulation module is a behavioral-level sandbox, and real voxel fits should use the toolbox. |
| Aghajari, Vinke & Ling (2020), *J Neurophysiol* 123(2):773–785, doi:10.1152/jn.00291.2019 | Cited by both sources above. The paper itself was not read. | Origin of the pSFT log-Gaussian model. |

The other literature is listed in `docs/PROTOCOL.md` §7 and labeled **verified** (read here) or **from memory, verify before citing**.

## 2. Novelty caution

Several existing lines of work are closely related: SF uncertainty in detection (classic psychophysics), feature-based attention gain and tuning (feature-similarity gain), expectation-related sharpening in V1, and statistical learning of stimulus distributions. The specific question tested here is whether pSFT shape tracks the *shape* of a cued SF prior: width, asymmetry and bimodality. We have not established that this question is new, and a full literature search is still outstanding (PROTOCOL §8).

## 3. Design decisions (defaults, all configurable)

- **Feature space**: log2 SF. The grid has 9 levels from 1 to 4 cpd in 0.25-octave steps, with the center at 2 cpd (log2 = 1).
- **Distributions**: discrete PMFs on the grid, from the families gaussian, uniform, skew-normal and two-Gaussian mixture. Two matching modes are available.
  - `mode_matched`: the peak sits at 2 cpd (the bimodal condition puts its two peaks symmetrically about 2 cpd).
  - `moment_matched`: broad, skewed and bimodal conditions are solved numerically to a shared mean and variance. Each result carries a `matched` flag that is set only if the residual is below 1e-6. On a bounded grid, uniform has a fixed variance and is reported as unmatched.
- **Sampling**: `probability` (i.i.d. draws) or `fixed_count` (largest-remainder counts, then shuffled). The two give different realized statistics and different sequential dependencies (see PROTOCOL §4).
- **Task**: CW/CCW tilt discrimination of a Gabor shown at a configurable eccentricity. Tilt is balanced within each SF level. An optional calibration block runs per-SF staircases under a uniform prior and then **freezes** tilt per SF, so difficulty is equated across SF without the staircase tracking the expectation effect.
- **Alternative paradigm** (`probe`): the target task is unchanged, and task-irrelevant probes are added in the mirrored hemifield. The probe SF multiset is identical in every condition, which follows the irrelevant-hemifield logic of Ramirez et al.
- **Display**: PsychoPy `GratingStim` in **pixel units**, with the pixel-per-degree conversion computed here from measured width, distance and resolution (exact atan), so no small-angle approximation is used. Configs marked `calibrated: false` are refused unless the run uses `--demo`.
- **Timing**: frame-counted presentation, measured refresh checked against the configured value, and dropped frames counted per trial.
- **Data**: one CSV row per trial, appended and fsynced, plus a JSON sidecar with the full config, seeds and versions. Resume works by regenerating the schedule from the seed and skipping trials already logged.

## 4. Architecture

```
sfu/
  calibration.py   monitor geometry, deg<->px, Nyquist check
  config.py        dataclass config, validation, JSON load/save
  distributions.py grid, PMFs, moment/mode matching, statistics
  stimulus.py      Gabor parameters; PsychoPy GratingStim factory; numpy preview
  schedule.py      seeds, Williams counterbalancing, sampling, balancing, probes
  adaptive.py      per-SF staircases (calibration block only)
  state.py         experiment state machine + checkpoint/resume
  responses.py     keyboard abstraction; simulated observer (headless)
  hardware.py      scanner trigger / event-marker hooks (null, log, serial)
  datalog.py       CSV/JSON writers
  runner.py        session driver, PsychoPy backend, headless backend, CLI
  gui.py           researcher interface (matplotlib widgets)
  analysis.py      realized-vs-intended validation, behavior, CIs, plots
  simulation.py    synthetic tuning profiles, model fits, parameter recovery
tests/             pytest
configs/           demo + pilot templates
```

## 5. Verification plan

pytest covers normalization, matched moments, reproducibility, balancing, SF bounds, complete logging, condition assignment and abort/resume. A headless demo produces example data, and diagnostic plots compare theoretical and realized distributions. If PsychoPy installs, an automated window smoke test runs without keyboard input. Hardware timing and a photodiode check of the display cannot be done here and are listed in the README.
