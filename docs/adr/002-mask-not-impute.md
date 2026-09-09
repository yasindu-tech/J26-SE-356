# ADR 002 — Mask missing modalities rather than impute them

**Status:** accepted
**Date:** 2026-09-02
**Deciders:** Team

## Context
Real screening settings never have all four modalities. Most published
multimodal work is evaluated on complete data, so the question of what to do
with an absent input is rarely forced.

## Decision
Late fusion takes an explicit **availability mask** (e.g. `[1,1,0,1]`). Absent
modalities are **excluded from the fusion computation**, not filled.

## Alternatives considered
- **Zero-filling** — rejected, and this is the important one. A zero is a value.
  Zero-filling teaches the model that "absent" resembles "normal", which in a
  screening context systematically under-flags exactly the patients whose data
  was hardest to collect.
- **Mean/median imputation** — rejected: fabricates evidence the clinician never
  collected, and produces falsely narrow confidence intervals.
- **Train a separate model per modality subset** — rejected: combinatorial, and
  each sub-model would be fit on too little data.

## Consequences
- Confidence intervals **widen honestly** when inputs are missing. This is a
  feature and must survive into the UI.
- Requires **degradation curves** per missing subset as a reported deliverable.
- Fusion weights must be learned per-patient rather than fixed.
