# Voice module — results

Aggregate numbers only. No person ids. Every number below is printed by a
script; nothing is hand-counted (CLAUDE.md section 3.6).

## Leakage audit (VOICE-14)

Reproduce (about 8.5 minutes on an M-series MacBook; outputs go to the
gitignored `models/voice/artifacts/`):

```bash
python models/voice/src/leakage_audit.py
```

Setup: UCI-470, 252 people (188 PD / 64 healthy), 756 recordings, 752 features.
L1 logistic regression with balanced class weights; scaler + `SelectKBest`
(mutual information) + model, with k ∈ {10, 25, 50, 100} and C ∈ {0.05, 0.1, 0.5}
tuned in an inner 3-fold CV; 5 outer folds; seed 42. Every metric is at
**person level** (mean of a person's 3 row probabilities), with 95% CIs from
1000 bootstrap resamples of people. Two runs gave byte-identical output.
Python 3.11.17, scikit-learn 1.9.1.

| Variant | Selection | Split | AUC [95% CI] | Balanced accuracy [95% CI] |
|---|---|---|---|---|
| A_full | leaky | row | 0.918 [0.876, 0.951] | 0.826 [0.769, 0.878] |
| A_sel | leaky | subject | 0.850 [0.791, 0.901] | 0.734 [0.664, 0.797] |
| A_split | infold | row | 0.910 [0.864, 0.948] | 0.826 [0.770, 0.874] |
| **B (honest)** | **infold** | **subject** | **0.812 [0.743, 0.874]** | **0.745 [0.675, 0.804]** |

| Paired AUC difference (same resampled people) | Value [95% CI] |
|---|---|
| **A_full − B (headline)** | **+0.105 [+0.064, +0.148]** |
| A_sel − B | +0.037 [+0.004, +0.073] |
| A_split − B | +0.097 [+0.061, +0.135] |

| Pure-noise check (mean ± SD over noise datasets) | In-fold AUC | Leaky AUC |
|---|---|---|
| ANOVA F scoring, 10 datasets | 0.526 ± 0.086 | 0.915 ± 0.021 |
| Mutual information scoring, 10 datasets | 0.501 ± 0.072 | 0.567 ± 0.061 |

### What it says

- The common evaluation (A_full) reports AUC 0.918. Evaluated honestly on
  unseen people (B), the same model family scores 0.812. The paired difference
  is +0.105, and its CI excludes zero.
- Most of that gap comes from **splitting by row instead of by person**
  (A_split − B = +0.097): when one person's recordings sit on both sides, the
  model partly recognises the person, not the disease.
- **Selection leakage alone** (A_sel − B = +0.037) is smaller, though its CI
  also excludes zero.
- The noise check shows the guard works: with in-fold selection, pure noise
  scores at chance for both scorers.

### Caveats

- With mutual-information scoring, leaky selection inflates pure noise much less
  (0.567) than ANOVA F does (0.915). On noise, mutual information scores most
  features as exactly 0, so its spurious picks carry little linear signal for the
  logistic model. The selection-leakage cost measured here is therefore specific
  to this scorer, and may be larger with a linear filter such as ANOVA F.
- In variant B the inner CV chose k = 100, the top of the grid, in 4 of 5 folds.
  A larger k might score differently; the grid was fixed in advance and not tuned
  to the result.
- One dataset, one recording protocol, people already diagnosed. These numbers
  are not evidence of early detection.

## Baseline ladder and screening PPV (VOICE-15)

Reproduce (about 1.5 minutes on an M-series MacBook; outputs go to the
gitignored `models/voice/artifacts/`):

```bash
python models/voice/src/baseline_ladder.py
```

Same people, folds, seed, scorer and bootstrap as the leakage audit, so every
rung is compared on the same resampled people. The L1 rung *is* variant B and
reproduces it exactly. Gender is a model input only in `sex_only` (decision D7).
The best single feature is chosen on each training fold alone. LightGBM uses the
same scaler and in-fold selector, with k and `num_leaves` ∈ {4, 8} tuned in the
inner CV. Threshold is a fixed 0.5 on balanced-class-weight probabilities, never
tuned on test people. LightGBM 4.7.0.

| Rung | AUC [95% CI] | Balanced accuracy [95% CI] | Sensitivity | Specificity |
|---|---|---|---|---|
| sex_only | 0.565 [0.487, 0.648] | 0.605 [0.538, 0.675] | 0.569 [0.500, 0.642] | 0.641 [0.526, 0.750] |
| best_single_feature | 0.740 [0.664, 0.810] | 0.650 [0.580, 0.716] | 0.628 [0.556, 0.698] | 0.672 [0.553, 0.785] |
| l1_logistic_all | 0.812 [0.743, 0.874] | 0.745 [0.675, 0.804] | 0.723 [0.658, 0.787] | 0.766 [0.656, 0.869] |
| **lightgbm_all** | **0.877 [0.825, 0.923]** | **0.778 [0.715, 0.834]** | 0.931 [0.892, 0.966] | 0.625 [0.500, 0.733] |

| Rung | PPV at 1% [95% CI] | PPV at 2% [95% CI] | PPV at 5% [95% CI] |
|---|---|---|---|
| sex_only | 0.016 [0.012, 0.023] | 0.031 [0.023, 0.046] | 0.077 [0.058, 0.110] |
| best_single_feature | 0.019 [0.014, 0.029] | 0.038 [0.027, 0.057] | 0.091 [0.067, 0.135] |
| l1_logistic_all | 0.030 [0.020, 0.054] | 0.059 [0.040, 0.103] | 0.140 [0.098, 0.229] |
| lightgbm_all | 0.025 [0.018, 0.034] | 0.048 [0.036, 0.067] | 0.116 [0.089, 0.155] |

| Paired AUC difference (same resampled people) | Value [95% CI] | CI above 0? |
|---|---|---|
| best_single_feature − sex_only | +0.175 [+0.078, +0.277] | yes |
| l1_logistic_all − sex_only | +0.247 [+0.155, +0.339] | yes |
| lightgbm_all − sex_only | +0.312 [+0.229, +0.397] | yes |
| l1_logistic_all − best_single_feature | +0.072 [+0.003, +0.139] | yes |
| lightgbm_all − best_single_feature | +0.137 [+0.068, +0.205] | yes |
| lightgbm_all − l1_logistic_all | +0.065 [+0.030, +0.106] | yes |

### What it says

- Every rung beats every rung below it on AUC. The honest LightGBM model is the
  best ranker (0.877), +0.065 over the L1 model.
- **Sex alone carries little.** Its AUC CI includes 0.5, so the sex–class
  correlation in UCI-470 is not what drives the full models.
- **One feature gets most of the way.** A single in-fold-chosen feature reaches
  0.740; the L1 model's margin over it is small (+0.072, lower bound +0.003).
- **Screening PPV is low for every rung.** At 1% prevalence, about 3 in 100
  people flagged by the best rung would have PD; at 5%, about 14 in 100. Output
  from this module can only prioritise a referral; it cannot confirm anything.
- **Higher AUC did not give higher PPV.** At the fixed 0.5 threshold LightGBM
  trades specificity (0.625) for sensitivity (0.931), and PPV at low prevalence
  depends mostly on specificity. L1 has the higher point PPV; the CIs overlap.

### Caveats

- The best single feature was not stable: `std_delta_delta_log_energy` (2 folds),
  `tqwt_entropy_log_dec_12` (2) and `std_delta_log_energy` (1).
- LightGBM chose the top of both grids (k = 100, `num_leaves` = 8) in all 5
  folds; L1 chose k = 100 in 4 of 5. Grids were fixed in advance and not widened
  after seeing the result.
- PPV assumes the sensitivity and specificity measured here transfer to a
  screening population. They come from people already diagnosed versus healthy
  controls, so real-world PPV is likely lower still.
- The 0.5 threshold is one operating point. A screening threshold would have to
  be chosen in-fold or on the D3 hold-out, never on these test people.

## Label-permutation control (VOICE-16)

Reproduce (about 10.5 minutes on an M-series MacBook; output goes to the
gitignored `models/voice/artifacts/`):

```bash
python models/voice/src/permutation_control.py
```

Labels are shuffled between people (each person keeps one label for all 3
recordings; 188 PD / 64 healthy kept), then the honest pipeline (in-fold
selection, person-level split, same grids and seed as above) is re-run. 10
permutations per model. The gate fails the run if the mean person-level AUC on
shuffled labels exceeds 0.55 (CLAUDE.md section 3.2).

| Model | Mean AUC on shuffled labels (n=10) | SD | Max | Gate (mean <= 0.55) | Real-label AUC |
|---|---|---|---|---|---|
| l1_logistic | 0.511 | 0.057 | 0.623 | pass | 0.812 |
| lightgbm | 0.506 | 0.058 | 0.604 | pass | 0.877 |

### What it says

- With the labels shuffled, both honest models score at chance. No route from
  test people into training was found.
- The real-label AUCs (0.812, 0.877) lie well above every one of the 20
  shuffled runs (highest 0.623).

### Caveats

- One shuffle alone reached 0.623, which is why the gate is on the mean of 10,
  not on a single run.
- 10 permutations are enough for a leak gate, not for a precise permutation
  p-value (the smallest possible would be 1/11).

## Sex-fairness breakdown (VOICE-17) — exploratory

Reproduce (about 1.5 minutes on an M-series MacBook; outputs go to the
gitignored `models/voice/artifacts/`):

```bash
python models/voice/src/sex_fairness.py
```

The honest models (same runs, folds and seed as the ladder) broken down by sex
code. Gender is **not** a model input (D7); it only splits the results. Each
group is bootstrapped on its own people; gaps are sex code 1 minus sex code 0,
with a stratified bootstrap CI. Threshold 0.5. The male/female meaning of the
codes is ⚠️ UNVERIFIED (not documented in the vault), so codes are reported as is.
**UCI-470 has no age column, so the age-band breakdown in the proposal cannot be
done on this dataset.**

| Group | People | PD | Healthy |
|---|---|---|---|
| sex code 0 | 122 | 81 | 41 |
| sex code 1 | 130 | 107 | 23 |

**L1 logistic** (overall AUC 0.812 [0.743, 0.874])

| Group | AUC | Balanced accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| sex code 0 | 0.820 [0.729, 0.898] | 0.736 [0.652, 0.807] | 0.667 [0.554, 0.756] | 0.805 [0.681, 0.917] |
| sex code 1 | 0.786 [0.671, 0.883] | 0.731 [0.624, 0.831] | 0.766 [0.685, 0.838] | 0.696 [0.500, 0.870] |
| gap (1 − 0) | −0.034 [−0.180, +0.097] | – | +0.100 [−0.026, +0.230] | −0.109 [−0.350, +0.104] |

**LightGBM** (overall AUC 0.877 [0.825, 0.923])

| Group | AUC | Balanced accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| sex code 0 | 0.878 [0.803, 0.938] | 0.810 [0.731, 0.886] | 0.889 [0.813, 0.947] | 0.732 [0.600, 0.865] |
| sex code 1 | 0.866 [0.788, 0.933] | 0.699 [0.602, 0.807] | 0.963 [0.920, 0.991] | **0.435 [0.250, 0.650]** |
| gap (1 − 0) | −0.012 [−0.115, +0.086] | – | +0.074 [+0.007, +0.153] | **−0.297 [−0.538, −0.037]** |

### What it says

- **Not driven by sex.** Within each sex both models still rank PD above healthy
  (every within-sex AUC CI is well above 0.5), so the sex–class correlation is
  not what the models learned.
- **Ranking is equal across sexes.** No AUC gap is distinguishable from zero for
  either model.
- **LightGBM's operating point is not equal.** At the 0.5 threshold it flags
  about 57% of healthy people with sex code 1 against about 27% with sex code 0
  (specificity 0.435 vs 0.732; gap CI excludes zero). The likely cause is that
  sex code 1 is 82% PD in this dataset, so the model's scores sit higher for
  that group. L1 shows the same direction but its gap CI includes zero.
- So equal AUC does **not** mean equal treatment: a single threshold would refer
  healthy people of one sex code far more often. Any screening threshold must
  be checked per sex before use.

### Caveats

- Exploratory: 122 and 130 people, and sex code 1 has only **23 healthy
  people**, so its specificity CI is very wide.
- Performance comes from people already diagnosed versus healthy controls; it
  says nothing about early-stage fairness.

## `predict_voice()` and demo (VOICE-18)

Reproduce (about 1.5 minutes; outputs go to the gitignored
`models/voice/artifacts/`: `demo_results.json`, `model_card.json`,
`voice_bundle.joblib`):

```bash
python models/voice/src/demo_voice.py                 # L1 logistic (default)
python models/voice/src/demo_voice.py --model lightgbm
```

Trained on the D3 training people only (201 people); the 51 hold-out people
(38 PD / 13 healthy) were never used in training, tuning or calibration. Chosen
k = 100, C = 0.1. Platt calibration with balanced class weights on out-of-fold
training scores; CI from a 50-model bootstrap ensemble over training people;
SHAP top 5 in log-odds (exact for the linear model).

| Hold-out check (51 people) | Value [95% CI] |
|---|---|
| AUC | 0.818 [0.651, 0.950] |
| Balanced accuracy | 0.727 [0.588, 0.864] |
| Sensitivity | 0.684 [0.525, 0.824] |
| Specificity | 0.769 [0.538, 1.000] |
| PPV at 1% / 2% / 5% | 0.029 / 0.057 / 0.135 |

The hold-out AUC agrees with the 5-fold CV evidence (0.812), but with 13
healthy people its CIs are very wide (the PPV upper bounds reach 1.0 when a
resample has no false positives). It is a sanity check, not the headline.

### Caveats

- The training-set sex-fairness note shows a specificity gap for **L1 as well**
  (0.85 vs 0.50 at the 0.5 threshold; gap CI −0.61 to −0.09). In the 5-fold CV
  over all people the L1 gap CI included zero, so the threshold issue is not
  LightGBM-only; it needs a per-sex check before any screening threshold is set.
- Calibrated probabilities assume a 50/50 prior. They are not the chance that
  a screened person has PD; use the PPV table for that.
- The quality gate checks the feature row only. PP1 has no audio, so recording
  quality (clipping, noise, length) is not checked yet. The raw-audio pipeline
  below adds that check for uploaded files.

## Raw-audio data and quality gate (VOICE-20, VOICE-21)

Reproduce (counts: seconds; features: about 15 seconds):

```bash
python scripts/count_voice_audio.py
python models/voice/src/build_audio_table.py
```

Figshare "Voice Samples for Patients with Parkinson's Disease and Healthy
Controls" (CC BY 4.0): one telephone "aaah" per person, WAV, 8 kHz mono.
81 people: 40 PD (mean age 67.0, range 43.1–85.3) and 41 healthy (mean age
47.7, range 18.0–79.0). **The healthy group is about 19 years younger**, so every
result below is checked against age.

| Quality gate | Recordings | Refused | Share refused |
|---|---|---|---|
| PD | 40 | 3 (clipping) | 0.075 |
| Healthy | 41 | 1 (background noise) | 0.024 |

77 people pass (37 PD / 40 healthy). Refusals are reported, not dropped
silently; the gate refused PD recordings more often here, which is checked
again as more data arrives.

## UCI-470 model on uploaded audio (VOICE-25)

Reproduce (about 40 seconds):

```bash
python models/voice/src/uci_transfer.py
```

The full UCI-470 model uses 24 features, and 22 of them (15 TQWT, 6 MFCC,
1 wavelet) cannot be recomputed from audio. So a UCI-470 model was retrained on
the **16 Baseline measures that Praat recomputes** (`uci_praat.py`: period mean
and SD, 5 jitter, 6 shimmer, 3 harmonicity), with the same honest pipeline
(person-level nested 5-fold CV, in-fold selection, k ∈ {4, 8, 16}, seed 42),
then tested on Figshare, which it never saw.

| Test | AUC [95% CI] | Balanced accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| UCI-470 CV, L1 logistic, 16 Praat measures | 0.731 [0.657, 0.801] | 0.652 [0.588, 0.721] | 0.601 [0.529, 0.667] | 0.703 [0.596, 0.820] |
| UCI-470 CV, LightGBM, 16 Praat measures | 0.771 [0.706, 0.833] | 0.680 [0.611, 0.745] | 0.766 [0.705, 0.823] | 0.594 [0.472, 0.705] |
| **Final L1 model on Figshare (external)** | **0.479 [0.351, 0.609]** | 0.430 [0.363, 0.498] | 0.811 [0.684, 0.921] | **0.050 [0.000, 0.128]** |
| Same, ages 43.1–79.0 only (34 PD / 23 healthy) | 0.432 [0.278, 0.597] | – | – | – |

Label-permutation control on UCI-470 (10 shuffles, L1): mean AUC 0.481, max
0.545, pass. External PPV at 1% prevalence: 0.009 [0.007, 0.010].

| Unit check: healthy median, Figshare ÷ UCI-470 | Ratio |
|---|---|
| period mean / SD | 1.16 / 1.32 |
| shimmer (6 measures) | 1.25 – 1.49 |
| harmonicity (3 measures) | 0.85 – 1.86 |
| **jitter local / absolute / rap / ppq5 / ddp** | **4.34 / 5.44 / 16.57 / 6.64 / 16.95** |

### What it says

- **On UCI-470 itself, 16 Praat measures carry real signal**: AUC 0.731 (L1)
  and 0.771 (LightGBM), below the 752-feature model's 0.812, as expected.
- **The UCI-470 model does not transfer to telephone audio.** On Figshare it
  scores at chance (0.479; CI includes 0.5). That is a cross-cohort drop of
  about 0.25 AUC.
- **Jitter is the reason it fails in a harmful direction.** Jitter from Figshare
  is 4 to 17 times higher than UCI-470's, so the model reads every voice as
  PD-like: it flags 95% of healthy people (specificity 0.050). Even inside
  UCI-470, rap jitter is only 0.15 times local jitter, where Praat usually gives
  about half, so the dataset's jitter was probably measured with other settings.
- **Explanations warn about it** (decision D13): jitter is never compared with
  UCI-470 voices for new audio, and a result that relies on it says so.
- **Keeping only measures that match does not help** (table below, post hoc:
  decided after seeing the result, so exploratory only).

| Features the UCI-470 model keeps (L1) | Inside UCI-470, nested CV | Figshare (external) |
|---|---|---|
| all 16 recomputable | 0.731 [0.657, 0.801] | 0.480 [0.351, 0.602] |
| 11 the unit check does not flag (no jitter) | 0.718 | 0.448 [0.321, 0.579] |
| 9 shimmer and HNR only | 0.740 | 0.500 [0.373, 0.634] |

  For the external column each model is tuned and fitted on all UCI-470
  people (`uci_transfer.subset_transfer`). Every version learns well inside
  UCI-470 and scores at chance on new recordings, so the missing and
  mismatched features are not the cause: what the model learns is tied to how
  UCI-470 was recorded. This is the cross-cohort drop.

### Caveats

- One external dataset, telephone quality. The transfer may be better on
  studio-quality audio closer to UCI-470's; there is no such dataset here yet.
- The UCI-470 audio is not public, so our Praat measures cannot be checked
  against theirs person by person; the unit check compares medians only.

## Raw-audio model (VOICE-22 to VOICE-24) — provisional, Figshare only

Reproduce (about 1 minute for the evidence, 15 seconds for the model):

```bash
python models/voice/src/audio_evidence.py
python models/voice/src/audio_model.py
```

56 named features from `audio_features.extract_voice_features` (Praat via
parselmouth, librosa), on the 77 quality-passed Figshare people. Same honest
pipeline: person-level nested 5-fold CV, in-fold scaling and selection
(k ∈ {5, 10, 20, 40}), seed 42, person-level bootstrap CIs (1000 resamples).

| Rung | AUC [95% CI] | Balanced accuracy [95% CI] | Sensitivity | Specificity |
|---|---|---|---|---|
| age + sex only | 0.866 [0.769, 0.948] | 0.830 [0.741, 0.910] | 0.811 [0.676, 0.923] | 0.850 [0.727, 0.950] |
| best single voice feature | 0.661 [0.542, 0.790] | 0.597 [0.490, 0.707] | 0.595 [0.438, 0.750] | 0.600 [0.450, 0.762] |
| L1 logistic, voice | 0.710 [0.586, 0.824] | 0.674 [0.571, 0.779] | 0.649 [0.483, 0.800] | 0.700 [0.553, 0.833] |
| LightGBM, voice | 0.716 [0.591, 0.828] | 0.651 [0.543, 0.766] | 0.703 [0.533, 0.857] | 0.600 [0.460, 0.756] |

| Rung | PPV at 1% | PPV at 2% | PPV at 5% |
|---|---|---|---|
| age + sex only | 0.052 [0.028, 0.142] | 0.099 [0.056, 0.251] | 0.222 [0.132, 0.464] |
| best single voice feature | 0.015 [0.010, 0.026] | 0.029 [0.019, 0.051] | 0.073 [0.048, 0.121] |
| L1 logistic, voice | 0.021 [0.013, 0.040] | 0.042 [0.026, 0.078] | 0.102 [0.066, 0.178] |
| LightGBM, voice | 0.017 [0.012, 0.030] | 0.035 [0.024, 0.059] | 0.085 [0.059, 0.139] |

| Paired AUC difference (same resampled people) | Value [95% CI] |
|---|---|
| best single voice feature − age + sex | −0.204 [−0.363, −0.042] |
| L1 logistic, voice − age + sex | −0.155 [−0.302, −0.009] |
| LightGBM, voice − age + sex | −0.149 [−0.304, −0.005] |

| Age-matched check (ages 43.1–79.0; 34 PD / 23 healthy) | AUC [95% CI] |
|---|---|
| age + sex only | 0.751 [0.590, 0.887] |
| L1 logistic, voice | 0.701 [0.556, 0.838] |
| LightGBM, voice | 0.730 [0.592, 0.860] |

Label-permutation control (10 shuffles): L1 mean 0.492 (max 0.637), LightGBM
mean 0.478 (max 0.612): both pass.

### What it says

- **Age beats voice on this dataset.** Age and sex alone reach AUC 0.866; both
  voice models score lower, and the paired CIs exclude zero. A headline number
  from this dataset that ignored age would be measuring age.
- **Voice still carries some signal beyond age.** Within the age range both
  classes share, the voice models keep AUC 0.70–0.73 with lower bounds above
  0.5, and shuffled labels score at chance.
- Trained and tested on the same kind of audio, this model works far better on
  telephone recordings than the UCI-470 model does (0.710 vs 0.479), but it has
  not been tested on any other dataset.

### Caveats

- One dataset, telephone audio, 77 people: provisional.
- The final model's hold-out check (16 people, 8 PD / 8 healthy) gave AUC
  0.547 [0.250, 0.857]: far too few people to tell anything. The CV numbers
  above are the evidence.
- The dataset gate (within-dataset AUC) cannot run with one dataset.
- The age-matched check still contains an age difference inside the range; it
  is a sanity check, not an age adjustment.

## Upload checks and bias of the raw-audio model — exploratory

Reproduce (about 1 minute):

```bash
python models/voice/src/audio_bias.py
```

### Checks on uploaded files

Before a score is given, an upload is refused, with a plain reason, if it is:

| Check | Threshold | Where |
|---|---|---|
| longer than one "aaah" (speech, wrong file) | > 30 s, checked before pitch tracking | `audio_features.early_problems` |
| almost silent | peak < 1% of full scale | `audio_features.early_problems` |
| steady vowel too short | < 1 s | quality gate |
| clipped | > 1% of samples | quality gate |
| background noise too loud | < 10 dB SNR (only when there is silence to measure) | quality gate |
| unsteady pitch | < 50% of the vowel voiced | quality gate |
| unlike every training voice | 3+ measures beyond the 1st–99th percentile span by half that span again | `audio_explain.far_outside_training` |

The last rule was chosen so it does not refuse real voices: on held-out
Figshare people (5-fold, training range from the other folds) it would refuse
1 of 40 healthy and 0 of 37 PD. Of the 16 hold-out test files, none is refused.

What the checks catch and miss, tried on altered copies of hold-out recordings:

- caught: a pure tone, a 40 s file, a near-silent file;
- **not caught: a male voice shifted up an octave.** It lands in the range of
  female voices; without knowing the person's sex this cannot be told apart. The
  card still shows its "outside the range seen in training" caution (4 measures);
- a faked loudness tremor (6 Hz) left the score unchanged (0.49 → 0.49,
  0.18 → 0.18): wobbling the volume does not push the score toward PD.

Deliberate voice changes in general cannot be detected reliably; the capture
protocol (a clinician present, a natural "aaah") is the main protection. When the
recording has no silence around the "aaah", background noise cannot be measured;
the result then says so.

### Bias by sex and age (77 people, out-of-fold L1 logistic)

Age and sex are not model inputs.

| Group | n PD / healthy | AUC [95% CI] |
|---|---|---|
| sex F | 18 / 25 | 0.65 |
| sex M | 19 / 15 | 0.84 |
| gap M − F | | +0.18 [−0.07, +0.41] |

| Score vs age, within class | Spearman r [95% CI] |
|---|---|
| healthy | −0.02 [−0.34, +0.31] |
| PD | +0.01 [−0.32, +0.35] |

| Healthy people referred (score ≥ 0.5), by age | Referred / n |
|---|---|
| ≤ 50 | 6 / 24 |
| 51–65 | 5 / 11 |
| > 65 | 1 / 5 |

### What it says

- **Within each class the score does not follow age** (r ≈ 0): older healthy
  people do not get higher scores than younger healthy people. The model is not
  simply reading age, although age still separates the classes in this dataset
  (see the ladder above).
- **The model looks weaker for women** (0.65 vs 0.84), with more false referrals
  of healthy women. The gap's CI includes zero, so it is not established, but it is
  shown on every result card ("Fairness by sex").
- **Other conditions that change the voice were not checked**: no dataset we hold
  labels colds, asthma, breathing, vocal-cord, throat or neck problems. The result
  card says on every score that these can raise it.

### Caveats

- 77 people; the age bands above 50 have 11 and 5 healthy people. Exploratory only.
- Sex codes are as recorded in Figshare's demographics file.

## XAI faithfulness and stability (raw-audio model) — exploratory

Reproduce (about 35 seconds):

```bash
python models/voice/src/xai_faithfulness.py
```

Same honest CV as the ladder (77 Figshare people, person-level 5 outer folds,
each tuned on its own training people). Follows the tapping module's FT-44
(deletion curve) and FT-45 (stability) checks. Figure:
`artifacts/xai_deletion_curve.png`.

### Are the reasons faithful? (deletion curve)

For each held-out person the features the model uses are set to the
training-fold median one by one, in that person's |SHAP| order, and compared
with the reverse order and 100 random orders of the same used features.

| Model | Pooled AUC after deleting, mean over 1–10 steps: SHAP / random / reverse | p | Mean change in log-odds: SHAP / random / reverse | p |
|---|---|---|---|---|
| **L1 logistic (used by the app)** | **0.521** / 0.576 / 0.611 | 0.010 | **0.863** / 0.711 / 0.536 | 0.010 |
| LightGBM | 0.592 / 0.645 / 0.671 | 0.020 | 2.632 / 1.684 / 1.039 | 0.010 |

Deleting only each person's top SHAP feature drops the L1 model's AUC from
0.710 to 0.525; a random used feature drops it to 0.658 on average.

### Are the reasons stable?

| Model | Single measures | Card groups (what the doctor sees) |
|---|---|---|
| L1, folds' importance ranking, mean Spearman (target 0.70) | 0.22 | 0.67 |
| L1, same top reason after a bootstrap refit (20 per fold) | 29% | 64% |
| L1, top-5 measures / top-3 groups overlap (Jaccard) | 0.22 | 0.55 |
| LightGBM, folds' ranking, mean Spearman | 0.17 | 0.79 |
| LightGBM, same top reason after refit | 14% | 49% |

Reasons the two fits share point the same way 98% of the time (L1; LightGBM 93%).
Overall the top groups are voice tone (MFCC), spectrum and pitch (L1).

### What it says

- **The explanations are faithful.** Removing what SHAP names first changes the
  score and the AUC faster than removing the same features at random (p ≤ 0.02,
  both models). The card shows what the model actually used.
- **The exact measure named is not stable; the group mostly is.** With 61–62
  training people and many near-duplicate measures (5 jitter, 6 shimmer, 26
  MFCC), a slightly different training set picks a different measure from the
  same family. The visible card groups measures (decision D12 and
  `voicePlain.ts`), which roughly doubles stability, but still falls just short
  of the 0.70 target for L1. Single measures in "technical details" should be
  read as one of several similar measures.
- For the linear model, faithfulness is close to guaranteed by construction;
  the check mainly confirms the implementation and the median baseline.
  Stability is the real finding, and it improves only with more training people.

### Caveats

- 77 people; one L1 fold kept a single feature, so the fold rankings compare
  few features (17 used by any fold).
- Deletion uses the training median, not the SHAP baseline (the training
  mean), so the curve is a check, not an exact replay of the SHAP values.
- Faithful means faithful to the model, not to the disease: these are the
  model's reasons, never causes.

## Age-balanced tests (raw-audio model) — exploratory

Reproduce (about 40 seconds):

```bash
python models/voice/src/age_balanced.py
```

The age-matched check above only re-scores a model that was trained with the
young healthy people. Here age is taken out of training too, two ways
(person-level nested CV, L1 logistic, seed 42, 1000 bootstrap resamples):

- **Age-adjusted voice**: inside each training fold every voice measure is
  regressed on age in the healthy training people (normal ageing), and that
  effect is subtracted for everyone (`AgeAdjuster`). Age is used only for the
  adjustment, never as a model input.
- **Same ages only**: people aged 43.1–79.0 (the range both classes cover),
  trained and tested inside it.

| Test | People (PD / healthy) | Age gap | Voice AUC [95% CI] | Age + sex only |
|---|---|---|---|---|
| Current model | 37 / 40 | 19.9 y | 0.710 [0.586, 0.824] | 0.866 [0.769, 0.948] |
| **Age-adjusted voice** | 37 / 40 | 19.9 y | **0.754 [0.639, 0.865]** | 0.866 |
| Same ages only, trained there | 34 / 23 | 9.9 y | 0.556 [0.403, 0.706] | 0.746 [0.581, 0.879] |

Age-adjusted minus current: +0.044 [−0.055, +0.135]. Same ages, voice minus
age + sex: −0.189 [−0.406, +0.030]. Label-permutation control for the
age-adjusted pipeline (10 shuffles): mean 0.521, max 0.626, pass.

### What it says

- **Voice carries signal beyond normal ageing.** With the healthy ageing trend
  removed from every measure, voice keeps AUC 0.754, lower bound 0.64. The
  current model is not just reading age through the voice.
- **But with same-age people only, voice cannot be shown to work.** Trained on
  about 46 people per fold, voice scores 0.556 and its CI includes 0.5. This
  mixes two things that cannot be separated here: fewer training people, and an
  age gap that is still 9.9 years inside the shared range.
- So: some voice signal, weak, and not cleanly separated from age on this
  dataset. More recordings of older healthy people are what would settle it.
- The age-adjusted model is not better by a margin we can show (CI includes
  0), and it would need the person's age at upload, so the app keeps the
  current model.

### Caveats

- The adjustment is linear and learned from healthy training people aged
  18–79; PD people up to 85 are slightly outside it.
- 77 people; 57 in the same-age test. Exploratory only.
