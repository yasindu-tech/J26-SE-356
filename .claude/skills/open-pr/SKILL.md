---
name: open-pr
description: >-
  Open a pull request in the J26-SE-356 (PD-XAI) repo, or explain the repo's
  branching/PR/release policy. This is the source of truth for that policy —
  CONTRIBUTING.md was removed from the repo, so this skill (plus CLAUDE.md §3
  for research-integrity rules) is where the process lives now. Always
  branches from and targets develop, and fills in the repo's PR template
  including the research-integrity checklist. Use when the user says "raise
  a PR", "open a PR", "push this as a PR", asks to submit work for review in
  that repo, or asks how branching/releases/commits work in J26-SE-356.
---

## Why this skill exists

J26-SE-356 (PD-XAI) used to document its branching model, commit conventions
and PR rules in a CONTRIBUTING.md file. That file was deliberately removed —
the team decided this knowledge should live as something an agent actively
applies, not a doc that goes stale. So this skill is now the only place the
full policy is written down. CLAUDE.md §5 has a short human-readable summary
and points here for detail. If you're asked to explain the process rather
than execute it, answer from this skill's content directly.

## Non-negotiables

1. **Base branch is always `develop`.** The only exceptions are `release/*`
   and `hotfix/*` branches, which target `main`. If you're not sure which
   applies, it's `develop`.
2. **Never commit directly to `main` or `develop`.** Both are protected on
   GitHub (`enforce_admins: true` — this applies to the repo owner too).
   Everything goes through a PR.
3. **The PR template gets filled in, not left blank.** Especially the
   research-integrity checklist — an unfilled checklist defeats the point of
   having one.
4. **Never tick a checklist box you haven't actually verified.** If you
   didn't check the code for a leakage pattern, don't tick "no leakage" —
   write "N/A —" with a reason instead, or leave it unticked with a note.

## Repo policy reference

### Branches

| Branch | Purpose | Protected | Merges from | Merges into |
|---|---|---|---|---|
| `main` | Submission / demo-ready. Tagged at each milestone. | ✅ | `release/*`, `hotfix/*` | — |
| `develop` | Integration. Default branch for PRs. | ✅ | `feature/*`, `fix/*` | `release/*` |
| `feature/<module>/<desc>` | New work | ❌ | `develop` | `develop` |
| `fix/<module>/<desc>` | Non-urgent bug fix | ❌ | `develop` | `develop` |
| `release/<version>` | Milestone stabilisation | ❌ | `develop` | `main` + `develop` |
| `hotfix/<desc>` | Urgent fix to a tagged submission | ❌ | `main` | `main` + `develop` |
| `docs/<desc>` | Docs-only | ❌ | `develop` | `develop` |
| `spike/<desc>` | Throwaway experiment. Never merged. | ❌ | any | — |

Keep branches short-lived — days, not weeks.

### Naming

`<type>/<module>/<short-kebab-description>`

Modules: `mri` · `gait` · `voice` · `tapping` · `fusion` · `mobile` ·
`desktop` · `backend` · `shared` · `repo`

```
✅ feature/mri/combat-inside-training-fold
✅ fix/voice/subject-wise-split-assertion
✅ docs/adr-003-reject-3d-transformers
❌ my-branch   ❌ yasindu-work   ❌ fix
```

### Commits — Conventional Commits

```
<type>(<scope>): <imperative summary, lower case, no trailing period>
```

| Type | Use for |
|---|---|
| `feat` | New capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `refactor` | Behaviour-preserving restructuring |
| `perf` | Performance |
| `chore` | Tooling, deps, config |
| `ci` | Workflow / pipeline changes |

Scopes match the module list above, plus `ci`. Never put PHI, subject IDs or
dataset contents in a commit message.

```
✅ feat(voice): move feature selection into inner CV loop
✅ fix(fusion): reject all-zero availability mask
❌ update stuff   ❌ WIP   ❌ Fixed bug.
```

### PR rules

| Rule | Value |
|---|---|
| Target branch | `develop` (only `release/*`/`hotfix/*` target `main`) |
| Approvals required | 0 — self-merge is allowed |
| CI | Must pass — this is the real gate |
| Conversations | All resolved before merge |
| Merge method | Squash and merge |
| Direct pushes to `main`/`develop` | Rejected by GitHub, for everyone |

**On self-merge:** GitHub disables the Approve button on your own PR — there
is no "allow self-approval" setting, which is why required approvals is 0,
not 1. What still protects the branch: a PR is still required, CI must pass
(currently: Mobile Build, Desktop App Build, Backend Build — see
`.github/workflows/ci.yml`), conversations must resolve, no force-push or
deletion. Still request a real review for anything touching
`models/common/`, `models/fusion/`, `services/backend/` or `packages/shared/`
— self-merge is permission, not encouragement.

Keep PRs under ~400 changed lines where practical; say why in the
description if it's larger.

### Releases

At each milestone (proposal demo, progress review, final submission):

```bash
git switch -c release/v1.0-proposal develop
# stabilise: docs, version bumps, no new features
# PR release/v1.0-proposal -> main, squash merge
git tag -a v1.0-proposal -m "Proposal presentation submission"
git push origin v1.0-proposal
# then PR main -> develop to carry back any release fixes
```

## Steps to open a PR

1. **Orient.** Run `git rev-parse --abbrev-ref HEAD`, `git status`, `git
   branch -a`, `git log --oneline -5` to see what branch you're on and
   whether there's already relevant work.

2. **Branch, if not already on a feature branch.** `git switch develop && git
   pull origin develop`, then `git switch -c <type>/<module>/<desc>` per the
   naming table above. If `develop` doesn't exist yet or can't be pulled,
   stop and tell the user — don't improvise a different base.

3. **Scan for data/PHI before committing anything.** Refuse and tell the user
   if the diff touches: anything under `data/raw|interim|processed|external`,
   files matching `.nii`, `.nii.gz`, `.dcm`, `.bval`, `.bvec`, `.wav`, `.mp4`,
   `.mov`, `.pkl`, `.joblib`, `.pt`, `.pth`, `.h5`, `.onnx`, or `.env`. This
   mirrors the `guard-no-data` check that used to run in CI (currently out of
   CI — see `.github/workflows/ci.yml` — so this manual scan matters more
   right now, not less).

4. **Commit** using Conventional Commits per the table above.

5. **Push.** `git push -u origin <branch>`. If this fails on credentials
   (`could not read Username for 'https://github.com'` or similar), stop and
   tell the user to run the push themselves — don't go hunting for tokens or
   try alternate auth methods.

6. **Fill the template and open the PR.** Read `.github/pull_request_template.md`,
   fill every section (see "Filling the checklist honestly" below), write it
   to a temp file, then:
   ```bash
   gh pr create --base develop --head <branch> --title "<type>(<scope>): <summary>" --body-file /tmp/pr-body.md
   ```
   If `gh` isn't available or isn't authenticated, print the filled body and
   the compare URL (`https://github.com/<owner>/<repo>/compare/develop...<branch>`)
   instead, and tell the user to paste it in manually.

7. **Report back.** Tell the user the PR URL (or the manual steps if `gh`
   wasn't available), and call out anything you marked "N/A" or left
   unticked so they can double-check your judgment before merging.

## Filling the checklist honestly

Don't tick a box because it's plausible — check for it. Concrete patterns to
grep for:

| Checklist item | What to actually check |
|---|---|
| Feature contract | Search the diff for `UPDRS`, `updrs`, `NHY`, `hoehn`, `medication`, `ledd`, `on_off`, `datscan`, `qsm` used as a column feeding a model's `fit`/`predict` — not just present anywhere (label/stratification use is fine) |
| Leakage | Find every `.fit(` / `.fit_transform(` call touching harmonisation, scalers, imputers, PCA, feature selection, resampling — confirm each is called *after* the train/test split, inside the fold, not on the full dataset |
| Splitting | Look for the split call (`GroupKFold`, `StratifiedGroupKFold`, or a manual splitter) and confirm it passes `groups=<subject id>`, not just shuffles rows |
| Metrics | Confirm balanced accuracy and AUC are reported (grep for `balanced_accuracy_score`, `roc_auc_score`), with a bootstrap CI computation nearby, not just raw `accuracy_score` |
| Baseline ladder | Confirm the PR's results include (or don't need to include, if it's not a modeling PR) age+sex, best single feature, best single modality alongside the fusion number |
| Missing modalities | Search for zero-fill (`fillna(0)`, `np.zeros_like`) on modality features — should be a mask, not a fill |
| Guards have red tests | Find the test file for whatever this PR adds; confirm at least one test asserts the *failure* case (e.g. asserts a `ValueError` when a subject straddles folds), not just the happy path |

If the PR is pure infra/docs/repo work with no model code, write "N/A — no
model code" once for the whole research-integrity section and move on — don't
force-fit these checks onto a `chore(repo)` commit.

## Example filled PR body

```markdown
## What and why

Moves neuroCombat harmonisation inside the training fold for the T1 pipeline.
Previously it was fit on the full dataset before the CV split, which leaks
site information into the test fold.

Closes #42

## Module

`mri`

## How to verify

\`\`\`bash
pytest models/mri/tests/test_harmonisation.py -v
\`\`\`

---

## Research-integrity checklist

- [x] **Feature contract** — no specialist-only inputs added; unchanged from before this PR
- [x] **Leakage** — `ComBat.fit()` now called inside `for train_idx, test_idx in cv.split(...)`, confirmed via `test_harmonisation.py::test_combat_never_sees_test_fold`
- [x] **Splitting** — unchanged, already subject-wise via `GroupKFold(groups=subject_id)`
- [ ] **Metrics** — N/A, this PR doesn't change metric reporting
- [ ] **Baseline ladder** — N/A, no new results reported in this PR
- [x] **Missing modalities** — unaffected by this change
- [x] **Guards have red tests** — added `test_combat_leaks_when_fit_before_split` which fails on the old code path

## Data and privacy

- [x] No files from `data/` committed
- [x] No PHI or subject identifiers anywhere in this PR
- [x] Notebook outputs cleared (n/a, no notebooks touched)
- [x] No `.env`, keys, tokens or credentials

## Standard checks

- [x] Base branch is `develop`
- [x] Rebased on latest `develop`
- [x] Tests pass locally — `pytest`
- [x] Lint clean — `ruff check .`
- [x] Docs updated in this PR if behaviour changed

---

## Numbers reported in this PR

none

## Notes for the reviewer

Old AUC (leaky) was 0.89 on the training-time eval — expect it to drop once
re-measured honestly post-fix. That's expected, not a regression.
```

## If invoked outside J26-SE-356

This skill's specifics (branch names, module list, checklist content) are
for this repo. If asked to open a PR somewhere else, fall back to a generic
flow: check the target repo's own CONTRIBUTING docs or PR template if one
exists, ask the user for the base branch if unclear, and use plain
Conventional Commits as a reasonable default.
