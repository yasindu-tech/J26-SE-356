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
- **No age column in UCI-470**, so the fairness analysis is by sex only; the
  age-band breakdown cannot be done on this dataset (see RESULTS.md, VOICE-17).
- **0 of 252 voice subjects have a PPMI scan** — this module is evaluated
  independently and is *not* in the fusion. Do not claim fusion here.
- End-to-end deep audio rejected at n=252.

## The headline deliverable
The **paired difference between the audited and unaudited pipelines**, with
bootstrap CIs — a number this literature does not currently report.

## PP1 scope

PP1 delivers an honest, tested voice-risk pipeline on the precomputed UCI-470
feature table, with the leakage audit as the headline result. No audio is
recorded or processed in PP1.

### Flow
UCI-470 feature table (`data/raw/UCI-470/pd_speech_features.csv`, gitignored)
→ feature-contract check → **person-level** split (StratifiedGroupKFold, seed 42)
→ scaler + feature selection **fitted inside each training fold** → L1 logistic
regression / LightGBM → Platt calibration + bootstrap-ensemble CI → SHAP top 5
→ result vector.

### Decisions
| # | Decision |
|---|---|
| D1 | Data is the UCI-470 table only. No audio and no feature extraction in PP1. |
| D2 | The unit is the person. A person's 3 rows get one score (mean of the 3 probabilities). Metrics and bootstraps resample people, not rows. |
| D3 | Final hold-out: 20% of people, chosen once, stratified by class and sex, seed 42. The demo model trains on the other 80%. Evidence numbers come from 5-fold person-level CV over all people. |
| D4 | Models: L1 logistic regression and LightGBM, both with in-fold selection, balanced class weights, seed 42 everywhere. |
| D5 | Result vector fields: `risk_score`, `ci_lower`, `ci_upper`, `triage_priority`, `availability_mask`, `shap_top5`, `sex_fairness_note`, `pipeline_version`, `quality_gate_passed`. |
| D6 | `triage_priority` is `prioritised_referral` or `not_prioritised`. Never diagnosis, positive, negative or cleared. |
| D7 | Gender is not an input to the main model. It is used only for the sex-only baseline and the fairness analysis. |
| D8 | Mobile recording and live feature extraction are PP2. The desktop result card waits for the desktop branch to merge. |

### Definition of done
1. `predict_voice(feature_row)` returns the full result vector for a held-out person.
2. Leakage audit: audited vs unaudited AUC with bootstrap 95% CIs and the paired difference.
3. Baseline ladder: sex only → best single feature → all features → LightGBM.
4. PPV at 1%, 2% and 5% prevalence.
5. Sex-fairness breakdown on the model card.
6. Red tests for the splitter, feature contract, in-fold selection and permutation control.
7. A demo command that scores five held-out people and writes the result JSON.
8. Desktop result card (or the demo command if the desktop branch is not merged).
9. One evidence slide: audit chart, ladder, sex breakdown, PPV.
10. No PHI, no data files, no secrets in commits.

### Out of scope for PP1
Audio capture, feature extraction from audio, the mobile app, and deployment.

### Setup
```bash
brew install python@3.11 libomp          # libomp is required by LightGBM on macOS
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r models/voice/requirements.txt
```
Place the dataset at `data/raw/UCI-470/pd_speech_features.csv`. The UCI download
wraps it in `pd_speech_features.rar` inside the zip; macOS `bsdtar -xf` extracts
it. The file has **two header rows** (a group row, then column names). Never copy
it outside `data/` — a `.csv` elsewhere is not gitignored.
