# models/mri — T1 + DTI fusion

**Owner:** D.G.Y. Sulochana (IT23267718)

**Question.** How much does adding DTI to T1 actually improve early-PD
discrimination — and do the explanations stay faithful and stable on a scanner
the model has never seen?

## Data — PPMI, verified 29 Aug 2026
| Cohort | N with T1 + DTI |
|---|---|
| PD | 210 |
| Prodromal | 82 |
| Healthy control | 76 |
| SWEDD (mimics) | 42 |
| **Total** | **410** |

## Pipeline
T1: registration → skull strip → spatial + intensity normalisation → PyRadiomics
DTI: motion + eddy correction → brain extraction → tensor fitting → FA/MD/AD/RD
→ **neuroCombat harmonisation (training fold only)** → LightGBM (3D CNN as
comparator) → Grad-CAM / M3D-CAM + SHAP → faithfulness testing → frozen
external validation on NTUA and BrainLat.

## Known problems
- **n≈210 PD is small for 3D deep learning** — radiomics + gradient boosting is
  the primary route; deep models are comparators only. 3D transformers are
  explicitly rejected (see `docs/adr/`).
- **Harmonisation leakage is the central risk.** Fitting neuroCombat before
  splitting contaminates everything downstream. Prior art: Saqib & Horovitz
  (2024) measured AUC 0.902 leaky vs **0.550** leak-free on T1 morphometry.
  Expect our honest number to be far below published figures.
- **Site is not in the downloadable PPMI tables** (excluded for PHI) — it must
  come from DICOM headers.
- FreeSurfer `recon-all` rejected: 8–12 h/subject.
