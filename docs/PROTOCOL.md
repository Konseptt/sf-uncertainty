# Protocol draft: SF uncertainty and population SF tuning

This is a working draft for discussion, not a final or preregistered protocol.

## Motivation

Ramirez, Wang and Ling (2025) measured voxel-wise population SF tuning (pSFT) in V1 to V3 while participants attended a low-SF (0.5 cpd) or high-SF (2 cpd) letter stream. Tuning was mapped with task-irrelevant band-limited noise (0.1 to 12 cpd) in the opposite hemifield. Attention shifted pSFT peaks toward the attended SF and sharpened bandwidth, and how much it did so depended on each voxel's distance from the attended SF.

Bloem, Bakst, McGuire and Ling (2025) varied the width of a spatial attention cue. The spatial profile of BOLD modulation broadened with cue width in V2 and V3, and gain decreased in V3. They modeled it with a generalized Gaussian that has a free shape parameter.

I want to ask the same question in feature space. When the upcoming SF is uncertain, does tuning in early visual cortex follow a single expected value or the full expected distribution? Changing the shape of the distribution matters here, because pSFT is not necessarily log-Gaussian, and only an asymmetric or bimodal prior can tell a log-Gaussian readout apart from one that matches the distribution.

## Conditions

All distributions are defined on a log2 grid from 1 to 4 cpd (9 levels, 0.25 octave steps, centered on 2 cpd).

| Condition | Manipulates | Default (moment-matched) |
|---|---|---|
| narrow | certainty | Gaussian, SD 0.20 octaves, mean 2 cpd |
| broad | uncertainty | Gaussian, SD 0.463 octaves; reference variance |
| uniform | maximal uncertainty | flat, SD 0.645 octaves; cannot be variance-matched |
| right_skewed | asymmetry toward high SF | skew-normal, mean 2 cpd, SD 0.463, skewness +0.41 |
| left_skewed | asymmetry toward low SF | mirror image, skewness −0.41 |
| bimodal | two expected SFs | Gaussian mixture, mean 2 cpd, SD 0.463, modes 1.41 and 2.83 cpd |

Means and variances are matched to within 1e-6. Entropy, skewness and kurtosis are reported but not matched, so conditions with equal variance still differ in entropy (bimodal 2.65 bits vs. broad 2.91 bits).

The two matching schemes treat skew differently.

- **Mode-matched**, peak at 2 cpd: the right-skewed condition puts 56% of trials above 2 cpd and the left-skewed condition 6%. This fits the idea of a distribution with a peak at 2 cpd and somewhat more probability at higher SFs.
- **Mean-matched**: the right-skewed condition peaks at 1.68 cpd and puts only 37% of trials above 2 cpd.

Which scheme to use for the main comparison is still open.

## Hypotheses

- **H0, point estimate.** Tuning depends only on the expected SF, so narrow, broad, skewed and bimodal conditions with equal means produce the same pSFT.
- **H1, uncertainty-scaled bandwidth.** Bandwidth grows with prior variance while tuning stays log-Gaussian.
- **H2, distribution-matched tuning.** The population profile follows the prior: skewed priors give asymmetric pSFT, and the bimodal prior gives a flattened or two-peaked pSFT.
- **H3, gain only.** Responses are scaled up at expected SFs without any change in tuning. At the voxel level, gain applied to a mixed population can look like a peak or bandwidth change, so H2 and H3 can only be separated with a pooling model.

## Sampling

With probability sampling, each trial is an independent draw. Trial history carries no information, but the realized proportions vary from block to block: with 108 trials, the realized SD varies by about ±0.05 octaves. With fixed-count sampling, the counts per level are set exactly and then shuffled. The realized distribution then matches the intended one to within 0.005 octaves, but the order is sampling without replacement, so late trials in a block are in principle predictable. I use fixed counts by default.

## Separating expectation, adaptation and attention

| Mechanism | Driven by | Control |
|---|---|---|
| Adaptation | stimulus history | identical probe sequence in every condition, trial-history regressors, and planned exposure-matched blocks that replay the same stimuli without a cue |
| Expectation | knowledge of the distribution | cue type (histogram, text, symbol, none), and cued vs. uncued blocks with the same stimuli |
| Feature-based attention | task relevance of SF | probes are task-irrelevant and in the opposite hemifield; the target task is the same in every condition |

When the distribution can only be learned from the stimuli, exposure and expectation are confounded. A cue without any exposure difference, and exposure without a cue, separate the two.

## Difficulty across SF

Orientation thresholds depend on SF, contrast, eccentricity and the number of visible cycles. I handle this in four ways:

1. A calibration block finds the tilt for each SF under a uniform prior, and that tilt is then fixed. Running the staircases through the main blocks would absorb the expectation effect itself.
2. Contrast is fixed and can be adjusted per participant.
3. The envelope can be fixed in degrees or in cycles. Fixing it in degrees confounds SF with the number of cycles. Fixing it in cycles confounds SF with stimulus size.
4. Conditions are compared at the same SF, and the expected probability of each trial is logged so it can be modeled within SF.

## Literature

Read in full:

- Ramirez, L.D., Wang, F., & Ling, S. (2025). Attention alters population spatial frequency tuning. *J Neurosci*. doi:10.1523/JNEUROSCI.0251-25.2025
- Bloem, I.M., Bakst, L., McGuire, J.T., & Ling, S. (2025). Dynamic estimation of the attentional field from visual cortical activity. *eLife* 14:RP104222. doi:10.7554/eLife.104222
- Ramirez, L.D., et al. (2026). The population spatial frequency toolbox. *J Open Research Software* 14(1):5. doi:10.5334/jors.610

Still to read and check before citing:

- Aghajari, Vinke & Ling (2020), *J Neurophysiol* 123(2):773–785 (pSFT method)
- Wiecek, Ramirez, Klimova & Ling (2026), *J Neurosci* (scale invariance of SF tuning)
- Davis, Kramer & Graham (1983), SF uncertainty in detection
- Blakemore & Campbell (1969), SF-selective adaptation
- Treue & Martínez-Trujillo (1999); Martinez-Trujillo & Treue (2004), feature-similarity gain
- Ling, Liu & Carrasco (2009), feature-based attention and tuning
- Kok, Jehee & de Lange (2012), expectation sharpening in V1
- Summerfield & de Lange (2014), expectation in perceptual decisions
- Fiser & Aslin (2001), visual statistical learning
- Girshick, Landy & Simoncelli (2011); Wei & Stocker (2015), priors and efficient coding
- Henriksson et al. (2008), SF tuning in human visual cortex

I have not done a systematic search yet, so I am not claiming the question is new.

## Open decisions

1. Mean-matched or mode-matched skew for the main comparison.
2. Variance, entropy, or both as the measure of uncertainty.
3. Explicit histogram cue or a prior learned from exposure.
4. Probe SF range and number of levels. In simulation with 13 probe levels and noise SD 0.1, BIC selected a non-Gaussian model for about 48% of Gaussian-generated curves, and a bimodal profile with a shallow dip was not recovered. Well-separated bimodal profiles were recovered every time. Testing H2 will need more probe levels, more data per voxel, or pooling.
5. Blocked or interleaved design, block length, and number of sessions.
6. Whether to keep the orientation task, or use a task orthogonal to SF in the scanner.
7. Per-SF contrast normalization as an alternative to tilt calibration.
8. Eye tracking, which is needed to confirm fixation with stimuli at 5° eccentricity.

## Current limitations

- Tested only headless and in a windowed run on a MacBook; no calibrated display, gamma correction or photodiode check yet.
- Behavioral results cannot show changes in neural tuning.
- No exposure-matched blocks, eye tracking or serial trigger input yet.
- The simulation uses static tuning curves, not a BOLD/HRF model.
