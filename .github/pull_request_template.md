<!--
Target branch must be `develop` (only release/* and hotfix/* target main).
Title must follow Conventional Commits — e.g. feat(mri): fit ComBat in-fold
Self-merge is allowed once CI is green. That means this checklist is the main
thing standing between us and a bad result in the final report. Tick it honestly.
-->

## What and why

<!-- One or two sentences: what changes, and what problem it solves. -->

Closes #

## Module

<!-- Delete all but one -->
`mri` · `gait` · `voice` · `tapping` · `fusion` · `mobile` · `desktop` · `backend` · `shared` · `docs` · `repo`

## How to verify

<!-- Commands a reviewer (or future you) can run. -->

```bash
```

---

## Research-integrity checklist

Tick, or write **N/A —** with a reason. Full detail in [CLAUDE.md §3](../CLAUDE.md).
**If this PR does not touch model code, write "N/A — no model code" once and move on.**

- [ ] **Feature contract** — no UPDRS / MDS-UPDRS, Hoehn & Yahr, clinician
      impression, medication or ON-OFF state, DaTscan or QSM used as a *model
      input*. (Label / target / stratification use is fine.)
- [ ] **Leakage** — harmonisation, feature selection, scalers, imputers, PCA and
      resampling are all fitted **inside the training fold only**
- [ ] **Splitting** — subject-wise, asserted in code. No subject on both sides.
- [ ] **Metrics** — balanced accuracy and AUC reported (not raw accuracy), with
      bootstrap 95% CIs; PPV quoted at screening prevalence
- [ ] **Baseline ladder** — compared against age+sex → best single feature →
      best single modality → fusion
- [ ] **Missing modalities** — masked out, never zero-filled or imputed
- [ ] **Guards have red tests** — anything enforcing a rule above has a test that
      *fails* when the rule is violated

## Data and privacy

- [ ] No files from `data/` committed
- [ ] No PHI or subject identifiers in code, logs, tests, fixtures, commit
      messages, or this PR description
- [ ] Notebook outputs cleared
- [ ] No `.env`, keys, tokens or credentials

## Standard checks

- [ ] Base branch is **`develop`**
- [ ] Rebased on latest `develop`
- [ ] Tests pass locally — `pytest` / `npm test`
- [ ] Lint clean — `ruff check .` / `npm run lint`
- [ ] Docs updated in this PR if behaviour changed

---

## Numbers reported in this PR

<!--
Any subject count, AUC or metric mentioned above must be reproducible from a
script or test — not from a notebook cell or memory. Cite the source.
Write "none" if this PR reports no figures.
-->

## Notes for the reviewer

<!-- Anything you're unsure about, or deliberately deferred. -->
