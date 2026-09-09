# models/tapping — finger tapping

**Owner:** Weerasinghe P.N.M. (IT23247918)

**Question.** Can smartphone-video finger tapping recover the kinematics
MDS-UPDRS Part III item 3.4 rates, well enough to contribute to screening?

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
