# Reference pipeline — UBU-PD-FT-Assessment

Source: https://github.com/arelraptor/UBU-PD-FT-Assessment
Commit read: bb844dfc279965be28d10ff3166204245b62404c (2026-05-04)
Dataset: HUBU-FIS, DOI 10.5281/zenodo.17738775
Read by: Weerasinghe P.N.M. (IT23247918), 2026-10-10

The official implementation released alongside the HUBU-FIS dataset.
This note records what it actually does, with line references, so the
novelty claims in the TAF and proposal are checkable rather than asserted.

## 1. Landmarks used

Three MediaPipe hand landmarks per frame (`lib/video_processing.py:126-134`):

- 8 — index fingertip
- 4 — thumb tip
- 0 — wrist

X, Y and Z are kept for each. Landmark 9 (middle-finger MCP) is not used.

## 2. Amplitude normalisation

    AMPLITUDE            = 3D distance(thumb tip 4, index tip 8)
    HAND_SIZE            = 3D distance(wrist 0, index tip 8)   # per frame
    NORMALIZED_AMPLITUDE = AMPLITUDE / HAND_SIZE

`lib/video_processing.py:145` and `:151`.

## 3. Smoothing

`lib/video_processing.py:154`

    df['SMOOTHED_AMPLITUDE'] = df['NORMALIZED_AMPLITUDE'].rolling(window=3, center=True).mean()

A centred 3-frame rolling mean, with backward then forward fill for the
edge NaNs. The window is fixed at 3 regardless of frame rate.

## 4. Tap detection

`lib/extract_features.py:115`

    peaks, _ = find_peaks(amplitude, distance=fps//3)

Only `distance` is set. No `prominence`, no `height`.

## 5. Sequence-effect features already present

`extract_classical_features` produces 27 feature columns. Exactly three
describe change over the run:

| Feature | Line | How it is computed |
|---|---|---|
| AMPLITUDE_SLOPE | :144 | `np.polyfit(x, y, 1)[0]` (:249) over SMOOTHED_AMPLITUDE against **frame index** |
| FREQUENCY_DROP | :147 | mean FREQUENCY of first half minus mean of second half |
| ACCELERATION_INCREASE | :150 | same half-split, on ACCELERATION |

None is computed per tap. The amplitude slope runs over frames, so it
mixes the within-tap oscillation with the across-tap decrement.

Note: the ACCELERATION_INCREASE comment says a positive value means an
increase, but the code computes first minus second half, so a positive
value means a decrease. Comment and code disagree.

## 6. Validation design

`lib/run_ml_classification.py:74`

    loo = LeaveOneOut()

Outer loop is leave-one-out over the rows of the diagnostic CSV — that is,
over individual **videos**. Inner loop is `GridSearchCV` (:86) scored on
MCC. Six classifiers: linear regression, kNN, decision tree, random
forest, AdaBoost, XGBoost.

There is no participant grouping anywhere in the file. Because 116 of the
118 participants contribute both hands, nearly every fold trains on one
hand of the test participant and tests on the other.

## 7. Prediction target and metrics

Target (`:72`): the UPDRS column, used directly as a multi-class label.
Metrics: MCC, F1, accuracy, acceptable accuracy (+/-1), precision, recall,
plus one-vs-rest weighted ROC AUC.

## 8. Explainability

None. `grep -rn "shap\|SHAP\|lime\|feature_importance" lib/` returns a
single hit, at `extract_features.py:390`, which is the word "shape" inside
an STFT docstring. There is no SHAP, no feature-importance reporting and
no faithfulness testing anywhere in the codebase.

## 9. How this module differs — mapped to the TAF novelty claims

| TAF novelty claim | Reference has | This module adds | Verdict |
|---|---|---|---|
| Sequence effect quantification | 3 coarse features: frame-level slope plus 2 half-split differences | Per-tap amplitude series; decrement onset, late/early ratio, min/initial ratio, hesitations, halts | **Narrowed** — the claim is the per-tap formulation, not decrement as such |
| Left-right asymmetry | Nothing; clips treated as independent rows | Participant-level L/R comparison, possible because the clip ID encodes participant and hand | **Supported** — the clinical ratings themselves show 42% of patients asymmetric vs 21% of controls |
| Video vs smartphone | Video only | Rate-feature transfer check on mPower | **Limited** — touchscreen capture cannot observe opening amplitude |
| Faithfulness-tested XAI | No XAI at all | SHAP plus deletion curve plus cross-fold rank stability | **Safe** — nothing to contest |

## 10. Methodological issues found in the reference

1. **The normalisation denominator moves, and shares a landmark with the
   numerator.** AMPLITUDE is |thumb(4) - index(8)| and HAND_SIZE is
   |wrist(0) - index(8)|, so landmark 8 appears in both. As the index
   finger flexes to meet the thumb, both shrink together, so the ratio
   partially cancels the oscillation being measured. HAND_SIZE is also
   recomputed every frame (`video_processing.py:145`), whereas a scale
   normaliser should be one value per clip. This module normalises by
   wrist(0) to middle-MCP(9), a segment that does not move when the
   fingers do.
2. **Tap rate is capped.** `distance=fps//3` allows at most about 3 taps
   per second. Healthy tapping reaches 4-6 Hz, so the fastest tappers
   lose taps — biasing exactly the group that should look fastest.
3. **No prominence threshold**, so small noise peaks are counted as taps.
4. **Validation is video-level**, so both hands of one participant
   routinely straddle train and test.
5. **Labels are treated as diagnosis.** The UPDRS column is a per-clip
   rating, not a diagnostic label: 27 control clips are rated 1 and 13
   patient clips are rated 0. Diagnostic group is recoverable from the
   clip ID prefix (CONTROL / ID), which the reference never uses.
6. **The smoothing window is frame-rate independent.** A fixed 3-frame
   window smooths a 60 fps clip half as much, in time, as a 30 fps clip.

## 11. What we reuse

- The MediaPipe Tasks landmark set (4, 8, 0) as a starting point, with
  landmark 9 added for normalisation.
- `find_peaks` as the tap detector, with prominence added and the
  distance bound relaxed.
- MCC as a reported metric, so numbers stay comparable with theirs.
- The published 27-feature list as baseline B2.

## 12. Open questions

- The repo README lists the associated paper as "pending". Check on
  release whether the published validation differs from the code.
- Is a 3-frame rolling mean enough to suppress landmark jitter at 30 fps?
  To be settled empirically in FT-19/FT-20.
