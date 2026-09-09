# CLAUDE.md — J26-SE-356

Guidance for Claude Code / AI agents working in this repository. Humans should
read this too: the research-integrity rules in section 3 are **not style
preferences, they are correctness requirements**, and violating them silently
invalidates published results.

---

## 1. What this project is

**Explainable Multimodal AI for Early Parkinson's Detection (PD-XAI)** —
SLIIT Research Project IT4010, group **J26-SE-356**.

A screening and triage platform that scores Parkinson's risk from **inputs a
non-specialist clinic can actually capture**: smartphone gait video, finger
tapping, voice, and (optionally) a routine T1 + DTI MRI.

**Critical framing — read this before writing any code or docs.**

> The platform is our **apparatus**, not our contribution. We build it because
> the research questions cannot be answered without an end-to-end system that
> runs on non-specialist inputs. What we submit is the **measurements** it
> produces: the accessibility cost, the cross-cohort drop, the faithfulness
> results, and the degradation curves.

This is **screening and triage, never diagnosis**. No code, comment, UI string,
API field or document may describe the output as a diagnosis. High risk means
*prioritised referral*; low risk means *not prioritised* — never *discharged*.

**Team and module ownership**

| Module | Owner | Student ID |
|---|---|---|
| T1 + DTI MRI fusion | D.G.Y. Sulochana (Yasindu) | IT23267718 |
| Gait (MediaPipe video) | Fernando W.S.N.J. | IT23259102 |
| Voice biomarkers | Ricky H.P.C. | IT23294134 |
| Finger tapping | Weerasinghe P.N.M. | IT23247918 |

Supervisor: Mr. Bimal Gunapala · Co-supervisor: Ms. Thilini Jayalath

---

## 2. Repository layout

```
apps/
  mobile/        React Native — patient-facing capture (gait, tapping, voice)
  desktop/       Electron — clinician workstation, MRI upload, results
services/
  backend/       FastAPI — orchestration, inference, storage
models/
  common/        Shared contracts, evaluation harness, preprocessing
  mri/           T1 + DTI            (owner: Yasindu)
  gait/          MediaPipe pose      (owner: Fernando)
  voice/         Acoustic features   (owner: Ricky)
  tapping/       Tap kinematics      (owner: Weerasinghe)
  fusion/        Late fusion + Level-2 cross-modal XAI
packages/
  shared/        TypeScript types shared by mobile, desktop and backend
docs/            Architecture, ADRs, research notes
data/            Local only. NEVER committed. See section 4.
scripts/         Dev and CI helper scripts
```

**Ownership rule:** you may freely change files inside your own `models/<your
module>/`. Changes to `models/common/`, `models/fusion/`, `services/backend/`
or `packages/shared/` affect everyone — those paths are covered by CODEOWNERS
and need the owning reviewer's approval.

---

## 3. Research-integrity rules — non-negotiable

These exist because each one, if broken, produces a number that looks better and
is wrong. Enforce them in code, not in good intentions.

### 3.1 The feature contract — no specialist-only inputs

**The following must NEVER be model input features:**

- UPDRS / MDS-UPDRS scores (any part)
- Hoehn & Yahr stage
- Clinician impressions or ratings
- Medication status, levodopa dose, ON/OFF state (a prescription reveals the diagnosis)
- DaTscan / DaT-SPECT and QSM derived values

They may be used **only** as labels, targets or stratification variables.
Predicting a diagnosis from the exam used to make that diagnosis is arithmetic,
not detection.

`models/common/contracts/` holds the runtime validator. It must be called before
any `fit()`. A model that trains without passing through the validator is a bug,
not a shortcut.

**Known and accepted overlap:** gait and finger-tapping features are *derived
from the same motor behaviours* MDS-UPDRS Part III rates. That is acceptable —
they are raw measurements, not a clinician's rating — but it must be stated
explicitly in any writeup before a reviewer states it for us.

### 3.2 Leakage — fit inside the fold, always

Anything **learned from data** goes inside the training fold:

- **Harmonisation** (neuroCombat) — fit on training fold only. Fitting on the
  full dataset before splitting contaminates every number downstream.
- **Feature selection** — inside the inner CV loop. With 752 voice features and
  252 subjects, selection on the full set decides the result.
- Scalers, imputers, PCA, resampling — all of it.

Two controls are mandatory and must fail the run loudly, not warn:

1. **Site-prediction gate** — if a classifier can still recover which
   scanner/site a sample came from after harmonisation, harmonisation failed.
2. **Label-permutation control** — shuffle labels, re-run; must score at chance.
   Above chance means something leaks.

> Prior art to be aware of: Saqib & Horovitz (2024) measured this exact penalty
> for T1 morphometry — AUC **0.902** with a leaky ComBat fit, **0.550** with the
> leak prevented. Assume our honest numbers will be far lower than the
> literature's headline figures. That gap is a finding, not a failure.

### 3.3 Splitting

**Subject-wise, always.** Multiple recordings, sessions or visits from one
person must never straddle a train/test boundary. Assert it in code; do not
trust a shuffle.

### 3.4 Metrics

- Report **balanced accuracy and AUC**, not raw accuracy. The voice dataset is
  2.9:1 imbalanced — predicting "PD" for everyone scores 75% and is useless.
- Report **PPV at realistic screening prevalence** (1%, 2%, 5%), not at
  research-set prevalence.
- Every headline metric gets a **bootstrap 95% confidence interval**.
- Compare against the full **baseline ladder**: age+sex → best single feature →
  best single modality → fusion. Fusion must beat all of them or it wasn't worth
  it, and we report that.

### 3.5 Missing modalities

Real screening always has something missing. Absent inputs are handled with an
**availability mask**, never zero-filled — zero-filling teaches the model that
"absent" means "normal", which in screening is dangerous. Confidence intervals
must widen honestly when inputs are missing.

### 3.6 Data provenance

- **Never treat a blank field as a negative.** An early PPMI audit read blank
  `MRIWDTI` as "no" and concluded there were 2 paired subjects; 82.7% of those
  records were simply blank and the true count was 410. Check null semantics
  before trusting any count.
- Every subject count quoted in a paper, slide or README must be reproducible by
  a script in `scripts/`. No hand-counted numbers.

---

## 4. Data handling — hard rules

- **Nothing from `data/` is ever committed.** PPMI, NTUA, BrainLat and UCI-470
  are all under data use agreements. A single committed scan or subject-level
  CSV is a DUA violation, and git history is not easily erasable.
- No PHI in code, comments, commit messages, issue titles, PR descriptions,
  logs, test fixtures or notebook outputs.
- **Clear notebook outputs before committing** (`nbstripout` is configured).
- Subject identifiers in logs must be hashed or pseudonymised.
- If you think you have committed data or PHI: **stop, do not push, tell the team
  lead.** History rewriting is possible before a push and painful after.

---

## 5. Working conventions

### Branching — see CONTRIBUTING.md for the full policy

- `main` — protected, release-ready. PR only.
- `develop` — protected, integration branch. PR only.
- `feature/<module>/<short-description>` — your working branches.

**Never commit directly to `main` or `develop`.** Both are protected; direct
pushes are rejected by GitHub for everyone, including the repo owner. Everything
goes through a pull request **into `develop`**.

**Self-merge is allowed** — you may merge your own PR once CI is green, without
waiting for a teammate. Required approvals is set to 0 deliberately (GitHub
disables the Approve button on your own PR, so any higher value would make
self-merge impossible). This means **CI and the PR checklist are the only
automated gates** — tick the research-integrity checklist honestly.

### Commits

Conventional Commits, with the module as scope:

```
feat(mri): add neuroCombat harmonisation fitted inside training fold
fix(voice): move feature selection into inner CV loop
docs(readme): document capture protocol
test(fusion): assert availability mask rejects all-zero input
```

Scopes: `mri`, `gait`, `voice`, `tapping`, `fusion`, `mobile`, `desktop`,
`backend`, `shared`, `docs`, `ci`, `repo`.

### Code style

- **Python** — 3.11+, `ruff` for lint and format, `mypy` where practical,
  `pytest` for tests. Type-hint public functions.
- **TypeScript** — strict mode on, `eslint` + `prettier`. No `any` without a
  comment explaining why.
- Prefer boring and readable over clever. This is a research codebase that four
  people and two supervisors will read.

### Tests

- Any function enforcing a rule in section 3 needs a test that proves it
  **fails** when the rule is violated. A validator with no failing-case test is
  not a validator.
- Run `pytest` for Python and `npm test` for JS before opening a PR.

---

## 6. Notes for AI agents specifically

- **Do not invent numbers.** Subject counts, AUCs, dataset sizes and citations
  must come from a verifiable source — a script, a dataset file, or a paper
  actually checked. If a figure is unverified, mark it `⚠️ UNVERIFIED` rather
  than stating it plainly.
- **Do not weaken a section-3 rule to make a test pass.** If a leakage guard
  fails, the pipeline is wrong; the guard is doing its job.
- **Do not add dependencies casually.** Large frameworks (3D transformers,
  end-to-end deep audio) have been deliberately rejected on sample-size grounds —
  check `docs/adr/` before reintroducing them.
- Ask before restructuring shared paths (`models/common`, `packages/shared`,
  `services/backend`) — four people depend on them.
- When you change behaviour, update the affected doc in the same PR.
