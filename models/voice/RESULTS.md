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
