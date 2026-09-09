# models/fusion — late fusion + Level-2 XAI

Combines per-modality scores into one calibrated risk estimate **without
assuming complete data**.

## The core design decision
An **availability mask** (e.g. `[1,1,0,1]`) tells the fusion layer which inputs
are real.

> **Missing modalities are masked out, never zero-filled** — zero-filling teaches
> the model that "absent" means "normal", which in a screening context is
> dangerous. Confidence intervals must widen honestly when inputs are missing.

## Outputs — three things, never one
1. Calibrated probability
2. Confidence interval that widens with missing inputs
3. Modality breakdown + Level-2 cross-modal explanation

## Honest scope
Fusion is trained and reported on the **332 subjects with both imaging and
gait**. Voice (0 paired) and tapping are evaluated as independent modules. The
architecture accepts them the moment paired data exists — but we do not claim
fusion we cannot demonstrate.

## Required deliverable
**Degradation curves** per missing subset: performance as each modality drops out.
