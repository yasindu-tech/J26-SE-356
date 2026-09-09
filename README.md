# PD-XAI — Explainable Multimodal AI for Early Parkinson's Detection

**SLIIT · IT4010 Research Project · Group J26-SE-356**

A screening and triage platform that estimates Parkinson's disease risk from
inputs a **non-specialist clinic can actually capture** — smartphone gait video,
finger tapping, voice — with an optional routine **T1 + DTI MRI** when a scanner
is available.

> **This is a screening and triage aid, not a diagnostic device.**
> High risk means prioritised referral. Low risk means not prioritised — never
> discharged. The system shortens the queue; it never closes the door.

---

## Research question

> How can an explainable, multimodal framework — combining smartphone-derived
> gait, finger-tapping and voice biomarkers with T1-weighted and DTI
> neuroimaging — deliver early Parkinson's screening that remains accurate,
> faithful and deployable in settings without specialist access?

The platform is our **apparatus**, not our contribution. What we submit is the
measurements it produces:

| # | Objective | The measurement it yields |
|---|---|---|
| 1 | Widen accessible input — four modules using only non-specialist inputs | The **accessibility cost** |
| 2 | Missing-modality-aware late fusion with an availability mask | **Degradation curves** per missing subset |
| 3 | XAI faithfulness testing as a mandatory pipeline stage | Deletion, stability and agreement scores |
| 4 | Train on PPMI, freeze, evaluate on unseen cohorts | The **cross-cohort drop**, attributed |

---

## Team

| Module | Owner | ID |
|---|---|---|
| T1 + DTI MRI fusion | D.G.Y. Sulochana | IT23267718 |
| Gait (markerless video) | Fernando W.S.N.J. | IT23259102 |
| Voice biomarkers | Ricky H.P.C. | IT23294134 |
| Finger tapping | Weerasinghe P.N.M. | IT23247918 |

Supervisor: **Mr. Bimal Gunapala** · Co-supervisor: **Ms. Thilini Jayalath**

---

## Architecture

```
        ┌──────────────── Input layer (what a clinic actually has) ────────────────┐
        │  Gait video      Voice audio     Finger tapping      MRI (T1 + DTI)      │
        │  (phone)         (phone mic)     (phone)             optional            │
        └───────┬──────────────┬───────────────┬───────────────────┬───────────────┘
                ▼              ▼               ▼                   ▼
          Gait model      Voice model     Tapping model        MRI model
          + XAI (L1)      + XAI (L1)      + XAI (L1)           + XAI (L1)
                └──────────────┴───────┬───────┴───────────────────┘
                                       ▼
                    Late fusion  ·  availability mask [1,1,0,1]
                    (missing modalities masked out, never zero-filled)
                                       ▼
                        Cross-modal explainability (Level 2)
                                       ▼
              Risk score  ·  modality breakdown  ·  clinical report
```

A **feature-contract validator** sits in front of every model and blocks
specialist-only inputs (UPDRS, Hoehn & Yahr, medication status) at runtime.
See [`CLAUDE.md` §3](./CLAUDE.md) for why.

---

## Repository layout

| Path | What lives here | Stack |
|---|---|---|
| `apps/mobile/` | Patient capture app | **React Native** |
| `apps/desktop/` | Clinician workstation | **Electron** |
| `services/backend/` | API, orchestration, inference | **FastAPI** (Python) |
| `models/common/` | Feature contract, evaluation harness | Python |
| `models/{mri,gait,voice,tapping}/` | Per-modality pipelines | Python |
| `models/fusion/` | Late fusion + Level-2 XAI | Python |
| `packages/shared/` | Shared TypeScript types | TypeScript |
| `docs/` | Architecture, ADRs, research notes | Markdown |
| `data/` | **Local only — never committed** | — |

---

## Getting started

**Prerequisites:** Node 20+, Python 3.11+, npm 10+. Mobile work additionally
needs Android Studio and/or Xcode.

```bash
git clone https://github.com/yasindu-tech/J26-SE-356.git
cd J26-SE-356

# JS workspaces (mobile, desktop, shared)
npm install

# Python environment (backend + models)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Run a piece of it:

```bash
npm run dev:mobile      # React Native / Metro
npm run dev:desktop     # Electron
npm run dev:backend     # FastAPI with reload on :8000
pytest                  # Python tests
npm test                # JS tests
```

Copy `.env.example` to `.env` and fill it in. **Never commit `.env`.**

---

## Data

This project uses **PPMI**, **NTUA**, **BrainLat** and **UCI-470**. All are
governed by data use agreements.

- Nothing in `data/` is ever committed — see `.gitignore`.
- Access must be requested per-dataset under your own DUA. Do not share
  downloads between team members outside the agreement's terms.
- No PHI in code, logs, commit messages, issues, PRs or notebook outputs.

| Dataset | Use | Verified counts |
|---|---|---|
| PPMI | Primary training cohort | 410 with T1 + DTI (210 PD / 82 prodromal / 76 HC / 42 SWEDD); 332 gait-imaging paired |
| UCI-470 | Voice | 252 subjects (188 PD / 64 HC), 756 recordings, 752 features, 2.9:1 |
| NTUA, BrainLat | External validation only — never trained on | — |

---

## Contributing

**`main` and `develop` are protected. Everything goes through a pull request.**

```bash
git switch develop && git pull
git switch -c feature/mri/combat-in-fold
# ... work, commit ...
git push -u origin feature/mri/combat-in-fold
# open a PR into develop
```

Read [CONTRIBUTING.md](./CONTRIBUTING.md) for the full branching model, commit
conventions and review requirements, and [CLAUDE.md](./CLAUDE.md) for the
research-integrity rules that PRs are reviewed against.

---

## Licence and academic integrity

Academic coursework for SLIIT IT4010. Not licensed for clinical use. **Not a
medical device.** No output of this system may be presented to a patient or
clinician as a diagnosis.
