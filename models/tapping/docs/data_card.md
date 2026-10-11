# HUBU-FIS finger-tapping data card

Task FT-09 · finger-tapping module (owner: Weerasinghe P.N.M., IT23247918) · as of 2026-10-11

This card describes the dataset the finger-tapping module trains and evaluates on,
how it is labelled and grouped, what it lacks, and how we handle it. Every count
comes from `scripts/count_hubu_fis.py` and its output,
[hubu_fis_counts.md](hubu_fis_counts.md) (machine-readable:
[hubu_fis_counts.json](hubu_fis_counts.json)). Do not edit counts here by hand;
re-run the script and update this card from its output.

## 1. Identity

| Field | Value |
|---|---|
| Name | HUBU-FIS Finger Tapping dataset |
| Source | Zenodo, DOI [10.5281/zenodo.17738775](https://doi.org/10.5281/zenodo.17738775), Version v1, published 2025-11-27 |
| Creators | Javier Amo-Salas, Álvaro García-Bustillo (Universidad de Burgos); Esther Cubo (Hospital Universitario de Burgos) |
| Funding (as written on the record) | "supported by the project PI19/00670 of the Ministerio de Ciencia, Innovación y Universidades, Instituto de Salud Carlos III, Spain" |
| File | `HUBU-FIS_FT.zip`, 10,805,859,055 bytes |
| md5 | `fc143d17f2d97eb2744c42908f49f482` (matches Zenodo; verified locally in FT-05 by Weerasinghe P.N.M.) |
| Label file | `fis_diagnostic.csv` (the dataset's own name), columns `ID` and `UPDRS` only; sha256 `1e15081cfc0aeab98b7e72c0a199e74bebc410bee4816f9f835294e8999a4677` |
| Related software | [arelraptor/UBU-PD-FT-Assessment](https://github.com/arelraptor/UBU-PD-FT-Assessment), linked from the record; our reference notes are based on commit `bb844dfc279965be28d10ff3166204245b62404c` ([reference-pipeline.md](reference-pipeline.md)) |

## 2. Contents

- **234 clips**: 226 `.mp4` and 8 `.MOV`. The label file and the video folder
  match 1:1.
- **At most one clip per hand per participant**, enforced by the loader (section
  5); the counts agree: 116 x 2 + 2 = 234 clips.
- **Groups:** CONTROL 85 clips, ID 149 clips.
- **Hands:** 117 left, 117 right.
- **Recording properties** (read with OpenCV 5.0.0; sizes are as displayed, see
  section 7):

| Property | CONTROL clips | ID clips |
|---|---|---|
| 1080x1920 | 65 | 120 |
| 1920x1080 | 10 | 19 |
| 1080x2400 | 0 | 4 |
| 1280x720 | 0 | 4 |
| 352x640 | 4 | 0 |
| 640x352 | 4 | 0 |
| 480x848 | 2 | 2 |
| portrait / landscape | 71 / 14 | 126 / 23 |
| 30 fps / 60 fps | 85 / 0 | 147 / 2 |
| `.mp4` / `.MOV` | 77 / 8 | 149 / 0 |
| length in s, min / median / max | 19.92 / 20.27 / 23.77 | 15.01 / 20.24 / 29.34 |

## 3. Labels

**What `UPDRS` means here.** We read `UPDRS` as a clinician's rating of
MDS-UPDRS item 3.4 (finger tapping) for that hand, on a 0-4 scale; the Zenodo
record itself does not describe the column. It is a **label**, never a model
input (a project rule, see `CLAUDE.md`), and it does not confirm or rule out
Parkinson's disease. The group (CONTROL or ID) comes from the clip ID, not from
the label file.

**Per clip.** 0: 71 (30.3%), 1: 99 (42.3%), 2: 38 (16.2%), 3: 26 (11.1%).
Rating 4 is never used.

| Clips | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| CONTROL | 58 | 27 | 0 | 0 |
| ID | 13 | 72 | 38 | 26 |

**Per participant (higher-rated hand).**

| Participants | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| CONTROL | 25 | 18 | 0 | 0 |
| ID | 2 | 33 | 22 | 18 |

**Group and rating are not the same thing.** 27 control clips are rated 1 and
13 patient clips are rated 0. The two candidate targets therefore disagree on
**18 CONTROL + 2 ID = 20 participants**, and on **27 + 13 = 40 clips**.

| Candidate target | Per clip | Ratio | Per participant | Ratio |
|---|---|---|---|---|
| CONTROL vs ID | 85 vs 149 | 1.75:1 | 43 vs 75 | 1.74:1 |
| UPDRS ≥ 1 vs 0 | 163 vs 71 | 2.30:1 | 91 vs 27 (higher hand) | 3.37:1 |

Both are imbalanced, so raw accuracy is never a headline metric; we report
balanced accuracy and AUROC with participant-level bootstrap 95% CIs (MCC as
well, for comparison with the reference pipeline).

**Left-right asymmetry in the rating** (participants with both hands): CONTROL
9 of 42 (21.4%, mean |R−L| 0.21); ID 31 of 74 (41.9%, mean 0.45).

## 4. Decisions made at the start of the module work (decided on 2026-10-11)

| # | Decision | Reasoning |
|---|---|---|
| 1 | **Grouping.** The participant key is group + participant number; every cross-validation fold is grouped by it. | Both hands of one person must never sit on both sides of a split, or the model is tested on a person it has already seen. |
| 2 | **Label scheme.** Keep UPDRS classes 0-3 as they are, with no merging. The primary target (CONTROL vs ID, or UPDRS ≥ 1 vs 0) is **not chosen yet** and will be decided with the supervisor. | The smallest class (rating 3) still has 26 clips and rating 4 never occurs, so there is nothing to merge; the target choice changes the research claim, so it is not ours alone. |
| 3 | **Missing demographics.** Age and sex are not available, so the age+sex rung of the baseline ladder is reported as **unavailable**, not silently skipped, until the FT-08 reply arrives. | The project's evaluation rules require every rung of the baseline ladder to be reported; a missing rung must be visible to a reviewer. |

## 5. Grouping in detail

- A clip ID has the form `<CONTROL|ID><number>_<DCHA|IZDA>`. The record describes
  "controls and patients"; our reading of the prefixes is that `CONTROL` is a
  healthy control and `ID` a patient. `DCHA` (*derecha*) is the right hand,
  `IZDA` (*izquierda*) the left.
- The participant key is the group plus the number, so `CONTROL<n>` and `ID<n>`
  with the same number are **different people**.
- **118 participants** (43 CONTROL, 75 ID); **116 with both hands** (42 CONTROL,
  74 ID); 2 with one hand (1 left only, 1 right only).
- Enforced in code, and each rule has a test that fails when it is broken:
  - `src/clip_id.py` accepts only the exact format (no whitespace, no other case,
    no extension) — `tests/test_clip_id.py`;
  - `src/hubu_fis.py` rejects duplicate IDs, the same participant and hand
    appearing twice, blank or out-of-range ratings, one person written two ways
    (`1` and `01`), two files for one clip, and any mismatch between labels and
    videos; `participant_table` gives a missing hand
    as empty, never 0 — `tests/test_hubu_fis.py`;
  - the audit script exits with an error and writes nothing when any rule
    fails — `tests/test_count_hubu_fis.py`.

## 6. What the dataset does not contain

- No age, sex, medication state, disease duration or rating 4.
- **Rater information is limited.** The record credits Dr. Gamez Leiva and
  Dr. Madrigal for the video assessments, but nothing per clip says who rated
  which clip or whether any clip was rated twice, so inter-rater agreement
  cannot be measured.
- Age, sex, medication state and disease duration were requested from the
  authors in FT-08 (email sent on **`<FT-08 SEND DATE — to fill in>`**). The reply is
  pending. Until then, Decision 3 applies.

## 7. Known confounds and quality notes

- **Recording settings found in only one group.** All 8 `.MOV` clips and all
  352x640 and 640x352 clips are CONTROL; all 1080x2400, 1280x720 and 60 fps clips
  are ID. **16 CONTROL and 8 ID clips** carry at least one such setting. A model
  could separate the groups by phone or camera instead of by tapping.
- **Rotation.** 193 clips are stored rotated with 90° metadata (41 are not). All
  sizes in this card are as displayed, after rotation, which is what OpenCV
  returns by default (auto-rotation on); the tracking step must keep this
  setting.
- **Planned mitigations (not done yet):** log these properties per clip in QC,
  and re-run every headline result without the flagged clips as a sensitivity
  analysis. We also plan to use resolution-independent features (landmark
  distances normalised by hand size) rather than pixel values.

## 8. Licence, citation and ethics

- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0);
  attribution is required (a CC BY 4.0 condition, not stated separately on the
  record). The record is open access.
- **Citation:** the record's description gives no citation instruction; we cite
  the DOI and the three creators listed above.
- **Ethics, as stated on the record:** recordings "were obtained following
  appropriate ethical approvals", and "participants provided informed consent
  for research use of their data".

## 9. Retention and handling plan

Existing repo rules (the project's data-handling rules, `data/README.md`, `.gitignore`):

- Raw data stays only under `data/raw/HUBU-FIS/`, which is gitignored and never
  committed.
- Tables that contain participant IDs (`hubu_fis_participants.csv`,
  `hubu_fis_clip_qc.csv`) are written only to `data/interim/` (gitignored).
- Committed outputs hold aggregates only. Participant IDs are never printed raw;
  anything flagged is shown as a short hash (a convention, not anonymisation:
  the ID space is small).

Proposed, **not yet confirmed with the supervisor**:

- Raw clips and the downloaded archive are never uploaded to third-party
  services (cloud notebooks, AI tools, shared drives).
- Derived landmark files (per-clip hand-landmark series, planned in FT-16) go
  to `data/interim/landmarks/` (gitignored) and keep only the landmark
  coordinates, frame times and the clip ID needed to join them to labels.
- Raw data and landmark files are kept until the end of the project, then
  deleted or archived as the supervisor directs.

## 10. Intended use and out of scope

- **Intended use:** research on screening and triage support from
  finger-tapping video, within the PD-XAI project.
- **Out of scope:** any clinical decision about an individual. No claim of
  clinical validity. UPDRS is never a model input. The participants are people
  already known to be patients or controls, so results do not show early
  detection.

## 11. Reproduction

From the repo root, with the raw data in place:

```bash
python scripts/count_hubu_fis.py --probe-video
```

`--probe-video` opens every clip and adds the recording-property tables
(sections 2 and 7). Without it, only the label and participant counts are
produced. A run without `--probe-video` refuses to overwrite counts that already
contain the probe tables (exit 1, nothing written); `--allow-drop-probe`
overrides this on purpose. The script stops with exit 1 and writes nothing if
any rule in section 5 fails.

## 12. As of

As of 2026-10-11. Numbers were generated by `scripts/count_hubu_fis.py` at the
commit that adds this card; last verified 2026-10-11.

## 13. Terms used in this card

- **PD-XAI, FT-xx:** PD-XAI is our group research project on explainable Parkinson's screening; FT-xx are numbered tasks in its finger-tapping work plan.
- **MDS-UPDRS item 3.4, rating 0-4:** the clinical scale's finger-tapping item, scored per hand from 0 (normal) to 4 (severe).
- **Balanced accuracy, AUROC:** accuracy averaged over the two classes so the larger class cannot dominate; area under the ROC curve, how well scores rank cases above controls (0.5 = chance).
- **MCC:** Matthews correlation coefficient, a single score from -1 to 1 that stays honest when classes are imbalanced.
- **Bootstrap 95% CI:** a range for a result, found by re-drawing participants with replacement many times and re-scoring.
- **Cross-validation fold:** one of several train/test splits of the data; every participant is in the test part of exactly one fold.
- **Baseline ladder, rung:** a fixed sequence of simple-to-complex comparison models; each step is a rung, the full model is compared against every rung, and whether it beats each one is reported.
- **QC:** quality control, per-clip checks on video properties and tracking before a clip is used.
- **Hand landmarks:** 21 points per hand per frame (fingertips, joints, wrist) found by the MediaPipe tracking model.
- **gitignored:** listed in `.gitignore`, so git never tracks or uploads the file.
