# models/gait — markerless video gait

**Owner:** Fernando W.S.N.J. (IT23259102)

**Question.** Do video-derived gait features retain the discriminative power of
sensor-derived ones?

## Data — PPMI gait substudy, verified
| Metric | Value |
|---|---|
| People with gait data | 385 |
| **Paired with imaging** | **332** ✅ |
| Roche smartphone app | 32 people (31 paired) |

## Pipeline
Video → frame extraction → MediaPipe Pose (33 landmarks) → coordinate
normalisation → temporal sequence → features (speed, cadence, stride, symmetry,
arm swing, variability) → model → SHAP.

## Known problems
- **The 332 paired subjects carry the entire fusion claim.** Say so explicitly
  in any writeup.
- Roche app data (31 paired) is far too small to train on — capture-protocol
  reference only.
- Gait features overlap with MDS-UPDRS Part III motor items by nature. That is
  acceptable (raw measurement, not clinician rating) but must be stated before a
  reviewer states it.
