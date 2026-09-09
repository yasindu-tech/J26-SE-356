# ADR 001 — Reject 3D transformers for the MRI module

**Status:** accepted
**Date:** 2026-09-02
**Deciders:** Yasindu (MRI module owner)

## Context
The MRI module has **410 subjects with paired T1 + DTI, of whom ~210 are PD**.
3D vision transformers are the current state of the art for volumetric imaging
and would be the obvious "modern" choice.

## Decision
Use **radiomics features + LightGBM** as the primary model, with a MONAI 3D CNN
as a comparator only. No 3D transformers.

## Alternatives considered
- **3D ViT / Swin-UNETR** — rejected. At n≈210 positive cases these overfit
  badly, and any resulting metric would be reporting noise. Choosing a model
  that fits the sample size is the honest call, not a compromise.
- **Full 3D CNN as primary** — rejected as primary, retained as comparator so we
  can report the difference rather than assert it.

## Consequences
- Our headline AUC will likely sit below deep-learning papers on this dataset.
  That is expected and is reported, not hidden.
- Radiomics features are directly interpretable, which makes the faithfulness
  testing in objective 3 tractable.
- Requires a defensible answer when asked "why not deep learning?" — the answer
  is sample size, and it is in this ADR.
