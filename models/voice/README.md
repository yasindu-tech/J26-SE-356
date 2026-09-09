# models/voice — leakage-audited voice biomarkers

**Owner:** Ricky H.P.C. (IT23294134)

**Question.** How much of the reported performance of voice-based PD detection
survives strictly subject-wise splitting with feature selection performed inside
each CV fold — and what is honest screening performance once prevalence and sex
confounding are accounted for?

## Data — UCI-470, verified
| Metric | Value |
|---|---|
| Subjects | 252 (188 PD / 64 HC) |
| Recordings | 756 (3 per subject) |
| Features | 752 |
| Class balance | 2.9 : 1 |

## Pipeline
Sustained vowel → quality gate + VAD → noise handling + normalisation → features
(jitter, shimmer, HNR, MFCC, TQWT, wavelet) → **feature selection INSIDE the CV
fold** → LightGBM / regularised linear → SHAP → fairness by sex and age band.

## Known problems
- **p ≫ n: 752 features, 252 subjects.** Selection must run inside each fold.
  Selecting on the full set first is the same leakage class as fitting ComBat
  outside the fold.
- **2.9:1 imbalance** — predicting "PD" for everyone scores 75%. Raw accuracy is
  meaningless here; report balanced accuracy and AUC.
- **Sex correlates with class** (41 HC / 81 PD vs 23 HC / 107 PD) — belongs in
  the fairness analysis and probably as a covariate.
- **0 of 252 voice subjects have a PPMI scan** — this module is evaluated
  independently and is *not* in the fusion. Do not claim fusion here.
- End-to-end deep audio rejected at n=252.

## The headline deliverable
The **paired difference between the audited and unaudited pipelines**, with
bootstrap CIs — a number this literature does not currently report.
