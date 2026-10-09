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
