# models/tapping — finger tapping

**Owner:** Weerasinghe P.N.M. (IT23247918)

**Question.** Can smartphone-video finger tapping recover the kinematics
MDS-UPDRS Part III item 3.4 rates, well enough to contribute to screening?

## Data — HUBU-FIS (Zenodo, DOI 10.5281/zenodo.17738775)
Raw files live in `data/raw/HUBU-FIS/` (`fis_diagnostic.csv` and `videos/`) and
are never committed. Every number below is printed by
`python scripts/count_hubu_fis.py --probe-video`; the full tables are in
[docs/hubu_fis_counts.md](docs/hubu_fis_counts.md) (machine-readable:
`docs/hubu_fis_counts.json`).

| Metric | Value |
|---|---|
| Clips | 234 (226 .mp4, 8 .MOV), labels and videos match 1:1 |
| Participants | 118 (43 CONTROL, 75 ID = patient); 116 with both hands |
| UPDRS 3.4 rating per clip, 0/1/2/3 | 71 / 99 / 38 / 26 (4 never used) |
| CONTROL vs ID | 85 / 149 clips (1.75:1); 43 / 75 participants (1.74:1) |
| UPDRS ≥ 1 vs 0 | 163 / 71 clips (2.30:1) |

- **UPDRS here is a clinician's rating of one hand's tapping, not a confirmation
  of disease**: 27 control clips are rated 1 and 13 patient clips are rated 0. It
  may be a label or a stratification variable, never a model input.
- **Both hands of a person share one participant key** (`src/clip_id.py`); folds
  must be grouped on it (`hubu_fis.participant_table`).
- **Device confound.** Some recording settings occur in only one group: all 8
  .MOV clips and the 352x640 / 640x352 clips are CONTROL; the 1080x2400,
  1280x720 and 60 fps clips are patients. Plan: per-clip QC logging, a
  sensitivity analysis without these clips, and resolution-independent features.
- **Orientation.** 193 clips are stored rotated with 90° metadata. Sizes are
  reported as displayed (after rotation), which is what the landmark model sees.
- **Keep the probe.** A run without `--probe-video` refuses to overwrite counts
  that contain the confound tables (exit 1, nothing written); pass
  `--allow-drop-probe` only if dropping them is intended.

## Pipeline
Video → MediaPipe Hands → landmark tracking → tap detection → kinematics
(rate, amplitude, decrement/fatigue, hesitations, arrhythmicity) → model → SHAP.

## Known problems
- Tapping features are **derived from the same behaviour** the MDS-UPDRS item
  rates. This is acceptable — raw kinematics, not a clinician's score — but it
  must be stated explicitly rather than discovered by a reviewer.
- Decrement and hesitation are the clinically distinctive signals; raw tap rate
  alone is weakly specific.
- Hand visibility and framing are the dominant capture-quality failure modes —
  gate on device.
