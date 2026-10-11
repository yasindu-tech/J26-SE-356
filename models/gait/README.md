# models/gait — video-based gait analysis for early PD detection

**Owner:** Fernando W.S.N.J. (IT23259102), GitHub `@nethal17`

**Question.** Can a plain walking video be turned into gait features that predict
PD gait severity (UPDRS gait 0, 1, 2–3), with an explanation a clinician can read?

**Demo goal.** Walking video in → MediaPipe pose → joint mapping → gait features
→ severity prediction → SHAP explanation, shown in the desktop app.
Fusion with the other modalities is out of scope for this component.

## Data
Raw data lives in `data/raw/` and is never committed (see `data/README.md`).

| Dataset | Used for | Notes |
|---|---|---|
| CARE-PD | Training and evaluation | 9 cohorts, SMPL-derived h36m 17-joint `.npz`, world coordinates in metres, Y up. fps 30 (PD-GaM 25). Labels (`UPDRS_GAIT`) come from the cohort `.pkl` files. |
| KOA-PD-NM | Demo input (real videos) | 191 MOV videos. Frame rate, resolution, clip length and severity scale are not documented, so they are checked before use. |

## Pipeline
Video → MediaPipe Pose (33 landmarks, `pose_world_landmarks`) → map to the h36m
17-joint order → heel strikes and gait cycles → gait features → XGBoost → SHAP.

- **Joint mapping.** Hips, knees, ankles, shoulders, elbows and wrists map
  directly. Pelvis is the hip midpoint, thorax the shoulder midpoint, spine the
  midpoint of the two. The y-axis is flipped to Y up.
- **Quality gate.** A video is rejected if landmark visibility is low, more than
  one person is present, or too few gait cycles are found.
- **Model.** XGBoost, compared against a majority-class baseline and logistic
  regression. Splits are `StratifiedGroupKFold` grouped by subject on pooled
  cohorts, plus leave-one-cohort-out. Metrics: macro-F1, balanced accuracy and a
  confusion matrix.

## Environment
```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r models/gait/requirements.txt
python models/gait/src/check_env.py
```
The pose model `pose_landmarker_heavy.task` goes in `models/gait/artifacts/`
(git-ignored). Run `check_env.py` to confirm it loads.

## Known limits
- **No demographics.** CARE-PD gives no age, sex or height, so the model cannot be
  compared against an age-and-sex baseline.
- **Domain gap.** CARE-PD poses are SMPL-derived, while the demo uses MediaPipe
  on raw video. Report this when quoting
  accuracy.
- **Merged severity class.** UPDRS gait 2 and 3 are one class. Report per-class
  results, not accuracy alone.
- Gait features overlap with MDS-UPDRS Part III motor items by nature. That is
  acceptable (raw measurement, not clinician rating) but must be stated before a
  reviewer states it.
