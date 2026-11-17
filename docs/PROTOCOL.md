# Research protocol (draft for discussion)

**Does uncertainty about upcoming spatial frequency reshape population SF tuning?**

Status: exploratory draft prepared for collaborative refinement with Dr. Sam Ling's lab. It is neither a validated fMRI protocol nor a pre-registration. The only implementation so far is a behavioral prototype.

## 1. Motivation

Ramirez, Wang & Ling (2025) measured voxel-wise population SF tuning (pSFT) in V1–V3 while participants attended to a low-SF (0.5 cpd) or high-SF (2 cpd) letter stream. pSFT was mapped with task-irrelevant band-limited noise (0.1–12 cpd) in the opposite hemifield. Attention produced *attractive* shifts in pSFT peak toward the attended SF and sharpened bandwidth. The size of the modulation varied with each voxel's dissimilarity from the attended SF.

Bloem, Bakst, McGuire & Ling (2025) varied the *width* of a spatial attentional cue. The spatial profile of BOLD modulation broadened with cue width (V2, V3) and its gain fell (V3). They fit it with a generalized Gaussian that has a free shape parameter.

These two results raise an open question in the **feature** domain. When the upcoming SF is uncertain, does tuning in early visual cortex track a single expected value (point estimate), or the full expected distribution (width, asymmetry, multimodality)? Dr. Ling suggested manipulating distribution **shape**, because pSFT need not be log-Gaussian. Only an asymmetric or bimodal prior can tell a log-Gaussian readout apart from a distribution-matched one.

## 2. Conditions

All conditions are defined on a log2-SF grid from 1 to 4 cpd (9 levels, 0.25-octave steps, center 2 cpd). They are listed in `sfu/distributions.py`, and the values below are reproduced by `scripts/diagnostics.py`.

| Condition | Manipulates | Moment-matched mode (default) |
|---|---|---|
| narrow | certainty | Gaussian, SD 0.20 oct, mean 2 cpd |
| broad | uncertainty | Gaussian, SD 0.463 oct (discrete); this is the reference variance |
| uniform | maximal uncertainty | flat; SD 0.645 oct, **not variance-matched** (no free parameter on a fixed grid) |
| right_skewed | asymmetry (tail toward high SF) | skew-normal solved to mean 2 cpd, SD 0.463; skewness +0.41 |
| left_skewed | asymmetry (tail toward low SF) | mirror; skewness −0.41 |
| bimodal | multimodality | two-Gaussian mixture solved to mean 2 cpd, SD 0.463; modes 1.41 and 2.83 cpd |

Each moment-matched condition is checked numerically: mean and variance residuals must be below 1e-6, or the condition is reported as unmatched. Entropy, skewness and kurtosis are reported but **not** matched. Matched-variance conditions still differ in entropy (bimodal 2.65 bits vs broad 2.91 bits), which matters if uncertainty is better indexed by entropy than by variance.

**Skew terminology.** In this repo "right-skewed" means positive statistical skewness, with the tail toward high SF. The two matching modes behave differently:

- **mode-matched** (peak at 2 cpd): right-skewed has P(SF > 2 cpd) = 0.56 vs left-skewed 0.06. This fits the intuitive description "more likely to be higher SF".
- **mean-matched**: right-skewed has its mode at 1.68 cpd and P(SF > 2 cpd) = 0.37, which is *lower* than the broad Gaussian's 0.40.

The two definitions therefore make opposite predictions about where the bulk of probability sits. Which comparison is the scientific one has to be decided before data collection (§8).

## 3. Competing hypotheses (predictions for probe-evoked pSFT)

- **H0 Point-estimate.** Tuning depends only on the expected (mean or modal) SF. The narrow, broad, skewed and bimodal conditions with equal means then give equal pSFT.
- **H1 Uncertainty-scaled bandwidth.** Bandwidth scales with prior variance (broad > narrow), but the shape remains log-Gaussian. This is the feature analogue of Bloem et al.'s cue-width broadening.
- **H2 Distribution-matched tuning.** The shape of the population profile follows the prior: skewed priors give asymmetric pSFT, and the bimodal prior gives a flattened or bimodal pSFT.
- **H3 Gain-only.** Responses scale up at expected SFs (feature-similarity gain) without any change in bandwidth or peak. At the voxel level, gain applied to a heterogeneous population can *look* like a shift or a change in bandwidth (see Ramirez et al.'s voxel strategies). Separating H2 from H3 therefore needs a model of how voxels pool their subpopulations.

## 4. Sampling: probability vs fixed count

- **probability** (i.i.d.): there is no sequential information, but realized proportions carry sampling error. At 108 trials, the 95% range of the realized SD is about ±0.05 octave (see `figures/diagnostics`). Realized statistics must be analyzed per block.
- **fixed_count** (counts set by largest-remainder rounding, then shuffled): realized proportions are exact up to rounding (SD error ≤ 0.005 octave). The sequence is sampled without replacement, though, so in principle it carries information near the end of a block, and rare levels have exact (sometimes zero) counts.

The default is fixed_count, so that conditions differ only in their intended distributions. One open question is whether to use i.i.d. sampling in the scanner (§8).

## 5. Separating attention, expectation and adaptation

| Mechanism | What drives it | Control |
|---|---|---|
| Sensory adaptation | Stimulus history: conditions with more 2-cpd stimuli adapt 2-cpd channels (cf. Blakemore & Campbell 1969) | (a) **Probe paradigm**: the probe SF stream is *physically identical* across conditions (same SFs, order, orientations and positions; tested in `test_probe_stream_identical_across_conditions`). (b) Trial-history regressors (SF of trials n−1…n−k). (c) **Exposure-matched control** (not yet implemented): replay a condition's exact target sequence with no cue and an uninformative instruction. Prior knowledge then differs while exposure stays equal. |
| Expectation / prior | Knowledge of the distribution (cue, learning) | Cue manipulation (`cue`: histogram / text / symbol / none). Compare cued vs yoked-uncued blocks with identical stimuli. |
| Feature-based attention | Task relevance of SFs | Probes are task-irrelevant and shown in the opposite hemifield (as in Ramirez et al.). The target task is the same in every condition. A future control would make the target task SF-independent, e.g. a contrast-change task. |

Stimulus exposure (adaptation) and prior knowledge (expectation) are confounded whenever the distribution is learned only from the stimuli. Cueing without exposure differences, and exposure without cueing, are the two dissociating manipulations.

## 6. Difficulty across SF

Orientation discrimination thresholds depend on SF, contrast sensitivity, eccentricity and the number of visible cycles. The prototype handles this in four ways:

1. An optional calibration block runs per-SF 2-down/1-up staircases under a uniform prior. The resulting tilt per SF is then **frozen** for the main blocks. A staircase that kept running during the main blocks would absorb the very expectation effect under study.
2. Contrast is fixed in Michelson units and can be adjusted per participant. Per-SF contrast normalization (CSF-based) is an alternative (§8).
3. The envelope is either fixed in degrees or fixed in cycles (`envelope`). Fixing it in degrees confounds SF with number of cycles. Fixing it in cycles confounds SF with size.
4. Analysis compares conditions **at the same SF**. The p_expected column lets expectation effects be modeled within each SF.

## 7. Literature

**Verified here** (full text or repository read during preparation):

- Ramirez, L.D., Wang, F., & Ling, S. (2025). Attention alters population spatial frequency tuning. *J Neurosci*. doi:10.1523/JNEUROSCI.0251-25.2025
- Bloem, I.M., Bakst, L., McGuire, J.T., & Ling, S. (2025). Dynamic estimation of the attentional field from visual cortical activity. *eLife* 14:RP104222. doi:10.7554/eLife.104222
- pSF-Toolbox: Ramirez et al. (2026), *J Open Research Software* 14(1):5, doi:10.5334/jors.610 (README only). GPL-3.0. No code is reused here.

**Cited by the papers above. Not read here; verify before citing:**

- Aghajari, S., Vinke, L.N., & Ling, S. (2020). *J Neurophysiol* 123(2):773–785, doi:10.1152/jn.00291.2019 (pSFT method).

**From background knowledge. Bibliographic details NOT verified in this session; check before citing:**

- Davis, Kramer & Graham (1983), uncertainty about spatial frequency in detection, *Perception & Psychophysics*.
- Blakemore & Campbell (1969), SF-selective adaptation, *J Physiol*.
- Treue & Martínez-Trujillo (1999), *Nature*; Martinez-Trujillo & Treue (2004), *Curr Biol*: feature-similarity gain.
- Ling, Liu & Carrasco (2009), *Vision Res*: feature-based attention, gain and tuning.
- Kok, Jehee & de Lange (2012), *Neuron*: expectation sharpens V1 representations.
- Summerfield & de Lange (2014), *Nat Rev Neurosci*: expectation in perceptual decision making.
- Fiser & Aslin (2001), *Psychol Sci*: statistical learning of visual structure.
- Girshick, Landy & Simoncelli (2011), *Nat Neurosci*; Wei & Stocker (2015), *Nat Neurosci*: priors and efficient coding shaping perception.
- Henriksson et al. (2008), *J Vis*: SF tuning in human retinotopic cortex.
- Peirce et al. (2019), *Behav Res Methods*: PsychoPy2.

A systematic search (SF uncertainty × fMRI, prior-shape learning, non-Gaussian tuning) has **not** been done. The question should not be described as novel until that search is complete.

## 8. Outstanding design decisions (for discussion)

1. Is the primary contrast mean-matched or mode-matched? This determines what "skewed" means (§2).
2. Should uncertainty be indexed by variance, entropy, or both? The two cannot be matched at the same time across these shapes.
3. How is the prior communicated: an explicit histogram cue, or learned from exposure? The choice affects how expectation and adaptation separate.
4. What is the probe SF range? Behavioral targets span 1–4 cpd. Estimating pSFT needs probes spanning ~0.1–12 cpd (as in Ramirez et al.), and the default probe range here is 0.5–8 cpd. Synthetic model recovery (`figures/simulation`) shows that **with 13 probe SFs and noise SD 0.1, BIC picks a non-Gaussian model for about 48% of Gaussian-generated curves, In the shallow-dip bimodal scenario, neither the Gaussian nor the spline fit recovers the dip (the BICs are nearly equal). Well-separated bimodal generators, by contrast, are recovered 100% of the time.** Testing H2 will need more probe levels, more data per voxel, or hierarchical pooling.
5. Should the design be blocked or interleaved, how long should blocks be for the prior to be learned, and should there be repeated sessions?
6. Should the orientation task be kept? A task that is itself SF-dependent couples attention to SF. An SF-orthogonal task (contrast decrement, fixation task) may be cleaner in the scanner.
7. Should per-SF contrast normalization be used instead of tilt calibration?
8. Should eye tracking be used? Covert attention at 5° eccentricity needs fixation control, and none is implemented.

## 9. Limitations of the current prototype

- It has only been tested headless and in a windowed autorespond smoke test on a MacBook. No calibrated display, photometer, gamma correction or photodiode validation has been done.
- Behavioral results cannot speak to neural tuning.
- There is no exposure-matched (yoked) block type yet, no eye tracking, and no serial-port trigger input (keyboard-emulated trigger only).
- The simulation is a static tuning-curve sandbox, not a BOLD/HRF model.
