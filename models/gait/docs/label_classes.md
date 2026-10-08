# Gait severity classes (GAIT-08, Check B)

Decision: the model predicts **three classes: 0, 1 and 2-3** (UPDRS gait score, with 2 and 3 merged).

Counts come from `python models/gait/src/count_labels.py`. The raw table is in
[`label_counts.csv`](label_counts.csv). Labels are from the cohort `.pkl` files (`UPDRS_GAIT`).
Walk copies named `_down0` to `_down4` are counted once. Walks shorter than 30 frames have no
skeleton in CARE-PD and are left out (2 in BMCLab, 1 in PD-GaM; all three have a score of 2 or 3).

## Subjects / walks per class

UPDRS gait score as recorded:

| Cohort | Score 0 | Score 1 | Score 2 | Score 3 |
|---|---|---|---|---|
| 3DGait | 16 / 24 | 26 / 42 | 12 / 14 | 5 / 10 |
| T-SDU-PD | 4 / 96 | 10 / 118 | 9 / 167 | - |
| BMCLab | 13 / 341 | 11 / 276 | 7 / 162 | - |
| PD-GaM | 29 / 783 | 28 / 635 | 20 / 248 | 2 / 34 |
| **All** | **62 / 1,244** | **75 / 1,071** | **48 / 591** | **7 / 44** |

Model classes:

| Cohort | Class 0 | Class 1 | Class 2-3 |
|---|---|---|---|
| 3DGait | 16 / 24 | 26 / 42 | 15 / 24 |
| T-SDU-PD | 4 / 96 | 10 / 118 | 9 / 167 |
| BMCLab | 13 / 341 | 11 / 276 | 7 / 162 |
| PD-GaM | 29 / 783 | 28 / 635 | 20 / 282 |
| **All** | **62 / 1,244** | **75 / 1,071** | **51 / 635** |

In total: 2,950 usable walks (42% / 36% / 22%) from 110 subjects. "Subjects" is the number of
subjects with at least one walk in that class, so the same person can appear in several
columns. Cohort totals are added together, assuming subject IDs in different cohorts are
different people (they are separate studies).

## Why three classes, and why 2 and 3 are merged

- Score 3 appears only in 3DGait and PD-GaM: **7 subjects and 44 walks** in total. That is below
  the plan's threshold of about 10 subjects, so it is too small to learn from or to test on.
- Merging 2 and 3 gives **51 subjects and 635 walks**, well above the threshold, so the
  fallback (0 versus 1 and above) is not needed. The check in `count_labels.py` applies this
  rule and prints "Decision: three-class".
- Scores 0 and 1 each have 60 or more subjects, so they stay separate.

## Things to keep in mind

- **Labels are per walk, not per person.** 62 of the 110 subjects have walks in more than one
  score (3DGait 16 of 43, T-SDU-PD 9 of 14, BMCLab 8 of 23, PD-GaM 29 of 30). I have not checked
  why. Whatever the cause, train/test splits must group by subject, and a subject has no single
  severity.
- **Walks per subject differ a lot** (about 57 in PD-GaM, 34 in BMCLab, 27 in T-SDU-PD, 2 in
  3DGait), so training must weight walks by subject or a few people will dominate.
- **Class 2-3 is the smallest** (22% of walks), so report macro-F1 and balanced accuracy, not
  plain accuracy.
- **Not comparable with the CARE-PD paper.** The published results use all four scores, so
  say on the slide that merged-class results are not directly comparable.
- The `medication` field in the label files is not a model input.
