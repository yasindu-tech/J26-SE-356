# models/gait — video-based gait analysis for early PD detection

**Owner:** Fernando W.S.N.J. (IT23259102), GitHub `@nethal17`

**Question.** Can a plain walking video be turned into gait features that predict
UPDRS gait severity (classes 0, 1, 2–3), with an explanation a clinician can read?
This is screening and triage, never diagnosis.

**Demo goal.** Walking video in → MediaPipe pose → joint mapping → gait features
→ severity prediction → SHAP explanation, shown in the desktop app.
Fusion with the other modalities is out of scope for this component.

## Data
Raw data lives in `data/raw/` and is never committed (see `data/README.md`).

| Dataset | Used for | Notes |
|---|---|---|
| CARE-PD | Training and evaluation | The four cohorts with UPDRS gait labels: 3DGait, T-SDU-PD, BMCLab, PD-GaM. SMPL-derived h36m 17-joint `.npz`, world coordinates in metres, Y up. fps 30 (PD-GaM 25). BMCLab's five `_down` copies count once. Labels (`UPDRS_GAIT`) come from the cohort `.pkl` files; how each file is read is recorded in `configs/carepd_format.toml`. |
| KOA-PD-NM | Demo input only (no severity labels) | MOV videos at 30 or 50 fps (read the rate from each file). `NM/015_NM_02.MOV` is empty. |

Counts (reproduce with the scripts below): 2,950 usable walks from 110 subjects;
2,738 walks from 109 subjects give the features.

## Pipeline (one module per step, in `src/`)
1. **Pose (video only)** — `pose_probe.py`: MediaPipe Pose Landmarker (heavy),
   world landmarks, one person tracked (`num_poses=1`).
2. **Joint mapping (video only)** — `mediapipe_mapping.py`: 33 landmarks → h36m
   17 joints. Axes `(x, -y, -z)`, a 180° rotation about x. A y flip alone is a
   mirror image that silently makes step length negative, so every mapped walk
   is checked (toe ahead of heel along the hip-derived facing in ≥ 90% of frames).
   Pelvis = hip midpoint, thorax = shoulder midpoint, spine between them.
3. **Usable frames (video only)** — person found and ≥ 80% of the leg landmarks
   visible; gaps up to 0.5 s are interpolated, longer gaps split the clip, and
   the longest stretch of at least 2 s is analysed. Turns are not cut out.
4. **Heel strikes and cycles** — `preprocess.py`, `cycles.py`: pelvis-centred,
   6 Hz zero-phase low-pass, heel strikes per foot; a stride counts only if its
   duration is plausible and the ankle really swings. At least 3 real strides
   (2 on short video clips), on both feet.
5. **Features** — `gait_extract.extract_features`, the single function used by
   training and the demo. Eight inputs: `stride_time_mean`, `cadence`,
   `stride_time_cv`, `step_length_norm`, `step_length_asymmetry`,
   `knee_flexion_range_deg`, `hip_flexion_range_deg`, `arm_swing_range_norm`.
   Lengths are divided by leg length. Every value is a median over the real
   strides. Leg length and stride count are returned for checks only, never as
   inputs. Step width and arm-swing asymmetry are computed but are not inputs.
   Values are rounded to 6 decimals in this one place, so the demo gives the model
   exactly the numbers of the training table.
6. **Rejection and flags** — a walk that cannot give the features is rejected
   with its reason (counted by cohort and class); nothing is imputed. A walk
   whose left and right stride times differ by more than 25% (with ≥ 3 strides
   per foot) is kept but flagged as a likely heel-strike detection problem.
7. **Splits** — `make_folds.py`: `StratifiedGroupKFold` by subject on the
   walk-level class, 5 folds × 5 seeds, saved and reused by every model. Every
   test fold must hold ≥ 3 subjects with walks of each class; no subject is on
   both sides.
8. **Model** — `train_xgboost.py`: XGBoost with fixed settings (150 trees, depth 3,
   no tuning) on the saved folds. The feature contract runs before every fit, the
   subject-wise split is asserted again at fit time, and a label-permutation
   control fails the run if it scores above chance. SHAP is computed on the
   out-of-fold models. A final model fitted on all subjects is saved for the demo
   as a bundle (`model_bundle.py`: model, input list, version; a reload is checked);
   it has seen every subject, so its own score is never reported.
9. **Checks on the score** — `evaluate_cohorts.py`: scores inside each cohort,
   leave-one-cohort-out (train on three cohorts, test on the fourth) and ablations
   (an input group or a cohort removed).
10. **Noise check** — `noise_check.py`: scores on test walks whose joints are
    jittered by multiples of the jitter measured on MediaPipe output.
11. **SHAP explanations** — `explain_shap.py`: global importance chart and the top
    three inputs for example walks, each in a plain-words sentence (`explain_words.py`),
    from the saved model.

## Evaluation (CLAUDE.md section 3.4)
- Headline: balanced accuracy and AUC (one-vs-rest), each with a bootstrap 95%
  CI resampled by subject. Secondary: macro-F1 (same interval) and the confusion matrix.
- Walk-level and subject-weighted scores (PD-GaM is 61% of the walks).
- Baseline ladder on the same folds: majority class, cohort-only, best single
  feature, logistic regression, XGBoost. Age and sex are not available in CARE-PD.
- PPV at screening prevalence is not reported: the output is a gait-severity
  grade, and no labelled cohort has a PD-vs-healthy label.

**Results (GAIT-19; reproduce with `train_xgboost.py`, XGBoost 3.2.0).** XGBoost:
balanced accuracy 0.553 (95% CI 0.503–0.602), AUC 0.750 (0.700–0.796); chance is
0.333. Shuffled labels: 0.328. Cohort only: 0.382. Logistic regression: 0.566, as
good as XGBoost on these data. Subject-weighted, XGBoost scores 0.506.

## Run
```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r models/gait/requirements.txt
python models/gait/src/check_env.py              # pose model loads
python models/gait/src/count_labels.py           # label counts per cohort and class
python models/gait/src/build_feature_table.py    # features of every CARE-PD walk
python models/gait/src/make_folds.py             # subject-wise folds (prints a fingerprint)
python models/gait/src/train_xgboost.py          # train, score, SHAP, save the model bundle
python models/gait/src/evaluate_cohorts.py       # by cohort, leave-one-cohort-out, ablations
python models/gait/src/check_mediapipe_mapping.py  # video pipeline on KOA-PD-NM clips
python models/gait/src/noise_check.py            # joint-noise check (needs the traces above)
python models/gait/src/explain_shap.py           # SHAP charts of the saved model
```
The pose model `pose_landmarker_heavy.task` goes in `models/gait/artifacts/`
(git-ignored). Outputs go to `models/gait/artifacts/` and are never committed.
Subject IDs are shown only as short hashes.

## Known limits
- **Small sample.** 109 subjects; class 2–3 is the most common class for only
  22 of them. Scores vary by about ±0.05 between folds.
- **Cohort signature.** The features identify the cohort well, while the cohort
  alone predicts the class only a little better than chance. Leave-one-cohort-out
  (GAIT-20) shows that transfer to a cohort the model has not seen is weak: balanced
  accuracy falls by 0.10 against cross-validation.
- **Frame resolution.** Stride times are whole frames (33–40 ms, about 3% of a
  stride), close to the size of `stride_time_cv` itself.
- **No demographics.** CARE-PD gives no age, sex or height, so there is no
  age-and-sex baseline.
- **Domain gap.** CARE-PD poses are SMPL-derived; the demo uses MediaPipe on raw
  video. Heel-strike detection is not validated on video and over-detects on
  noisy ankle traces (the left/right flag catches some cases). MediaPipe output
  differs slightly between machines.
- **Demo recording.** One straight walk, side-on, 2–4 m from the camera, whole
  body in frame, at least 10 s. Only one person is tracked; a second person in
  the frame is not detected or rejected.
- **Severity grading, not PD vs healthy.** The labels grade gait impairment (UPDRS gait) in the
  four labelled cohorts, and none of them stores a healthy/PD group. The model therefore ranks
  how impaired a walk is; it does not separate people with PD from people without. The closest
  thing to an early sign here is class 0 against class 1 (subtle impairment against none).
- **Merged severity class.** UPDRS gait 2 and 3 are one class. Report per-class
  results, not accuracy alone.
- Gait features overlap with MDS-UPDRS Part III motor items by nature. That is
  acceptable (raw measurement, not clinician rating) but must be stated before a
  reviewer states it.
  