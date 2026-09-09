## What and why

<!-- One or two sentences. What changes, and what problem it solves. -->

Closes #

## Module

<!-- Tick one -->
- [ ] `mri` · [ ] `gait` · [ ] `voice` · [ ] `tapping` · [ ] `fusion`
- [ ] `mobile` · [ ] `desktop` · [ ] `backend` · [ ] `shared` · [ ] `docs` / `repo`

## How to verify

<!-- Commands a reviewer can run, or steps to click through. -->

```bash
```

## Research-integrity checklist

Tick or write **N/A** with a reason. These are what our marks depend on — see
[CLAUDE.md §3](../CLAUDE.md).

- [ ] **Feature contract** — no UPDRS, Hoehn & Yahr, clinician impression,
      medication status, DaTscan or QSM used as a *model input*
- [ ] **Leakage** — harmonisation, feature selection, scaling and resampling all
      fitted **inside the training fold only**
- [ ] **Splitting** — subject-wise; no subject appears on both sides of a split
- [ ] **Metrics** — balanced accuracy / AUC reported (not raw accuracy), with
      bootstrap CIs
- [ ] **Missing modalities** — masked, not zero-filled
- [ ] Any rule-enforcing code has a test that **fails when the rule is violated**

## Data and privacy

- [ ] No files from `data/` committed
- [ ] No PHI or subject identifiers in code, logs, tests, commit messages or this PR
- [ ] Notebook outputs cleared
- [ ] No `.env`, keys or credentials

## Standard checks

- [ ] Rebased on latest `develop`
- [ ] Tests pass locally (`pytest` / `npm test`)
- [ ] Lint clean (`ruff check .` / `npm run lint`)
- [ ] Docs updated in this PR if behaviour changed

## Notes for the reviewer

<!-- Anything you're unsure about, or want a second opinion on. -->
