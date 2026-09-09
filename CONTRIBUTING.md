# Contributing — J26-SE-356

Four people, one integration, one grade. These rules exist so that nobody's
work-in-progress can break somebody else's demo the night before a review.

**The one rule that matters most:**

> ### 🔒 Nobody commits directly to `main` or `develop`. Ever. Including the repo owner.
> Both branches are protected on GitHub. Direct pushes are **rejected by the
> server**, not by convention. Every change arrives via a pull request into
> `develop`.
>
> You **may merge your own PR** once CI is green — you don't need to wait for a
> teammate. The PR requirement stays; the approval requirement doesn't.

---

## 1. Branching model

A trimmed Git Flow — enough structure for a four-person team, not more.

```
main ──────●─────────────────────────●────────────────●──────►   protected · release-ready
            \                       /                /
develop ─────●────●────●────●──────●────────●───────●─────────►  protected · integration
              \      /      \                \      /
feature/... ───●────●        ●────●           ●────●             your work happens here
```

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

### Naming

`<type>/<module>/<short-kebab-description>`

Modules: `mri` · `gait` · `voice` · `tapping` · `fusion` · `mobile` · `desktop`
· `backend` · `shared` · `repo`

```
✅ feature/mri/combat-inside-training-fold
✅ feature/mobile/qr-pairing-flow
✅ fix/voice/subject-wise-split-assertion
✅ docs/adr-003-reject-3d-transformers
❌ my-branch          ❌ yasindu-work         ❌ fix
```

Keep branches **short-lived** — days, not weeks. A branch open for three weeks
is a merge conflict with a countdown timer.

---

## 2. The workflow

```bash
# 1. Start from current develop
git switch develop
git pull origin develop

# 2. Branch
git switch -c feature/mri/combat-inside-training-fold

# 3. Work. Commit in small, meaningful steps.
git add models/mri/src/harmonisation.py
git commit -m "feat(mri): fit neuroCombat inside the training fold"

# 4. Keep up to date — rebase, don't merge, while the branch is yours alone
git fetch origin
git rebase origin/develop

# 5. Push and open a PR into develop
git push -u origin feature/mri/combat-inside-training-fold
```

**Rebase your own branch; never rebase a shared one.** If someone else has
pulled your branch, merge instead.

---

## 3. Commits

[Conventional Commits](https://www.conventionalcommits.org/):

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

```
✅ feat(voice): move feature selection into inner CV loop
✅ fix(fusion): reject all-zero availability mask
✅ test(mri): assert harmonisation never sees test-fold subjects
❌ update stuff       ❌ WIP        ❌ Fixed bug.
```

**Never put PHI, subject IDs or dataset contents in a commit message.**

---

## 4. Pull requests

### Before you open one

- [ ] Rebased on latest `develop`
- [ ] `pytest` passes (Python) / `npm test` passes (JS)
- [ ] Lint clean — `ruff check .` / `npm run lint`
- [ ] Notebook outputs cleared
- [ ] No files from `data/`, no `.env`, no secrets, no PHI
- [ ] Docs updated in the same PR if behaviour changed

### Rules

| Rule | Value |
|---|---|
| Target branch | **`develop`** (only `release/*` and `hotfix/*` target `main`) |
| Approvals required | **0** — self-merge is allowed |
| CI | **Must pass** — merging is blocked otherwise. This is the real gate. |
| Conversations | All resolved before merge |
| Merge method | **Squash and merge**, so `develop` history stays one-commit-per-PR |
| Branch deletion | Automatic after merge |
| Direct pushes to `main`/`develop` | **Rejected by GitHub**, for everyone including the owner |

### On self-merge

**You can merge your own PR once CI is green.** You do not need to wait for a
teammate. With four people on different modules and different schedules,
blocking on peer review would cost more than it catches.

One technical note, because it surprises people: **GitHub does not allow you to
approve your own pull request** — the Approve button is disabled on PRs you
authored. There is no "allow self-approval" setting. So required approvals is
set to **0**; any value ≥ 1 would make self-merge impossible regardless of what
else is configured.

**What still protects you with 0 approvals:**

- A PR is still required — nobody pushes to `main` or `develop` directly
- CI must pass: lint, tests, and the guards that reject committed datasets,
  model artefacts, `.env` files and un-cleared notebook outputs
- Conversations must be resolved
- No force pushes, no branch deletion
- The PR template's research-integrity checklist — **you tick it yourself, so
  tick it honestly.** That checklist is the thing standing between us and a
  leaky pipeline in the final report.

**Still ask for a review when it matters** — anything touching
`models/common/`, `models/fusion/`, `services/backend/` or `packages/shared/`
affects all four of us. Self-merge is permission, not encouragement.

### Size

Keep PRs reviewable — **under ~400 changed lines** where you can. If a PR is
large, say why in the description. Nobody reviews a 2,000-line PR properly; they
approve it, which is worse than not reviewing it.

### Reviewing

CODEOWNERS auto-requests the right person. When reviewing, check specifically:

1. **Section 3 of [CLAUDE.md](./CLAUDE.md)** — feature contract, leakage,
   subject-wise splitting, metrics. This is what our marks depend on.
2. No data, secrets or PHI added.
3. Tests exist for anything that enforces a rule.
4. Naming and structure match the module conventions.

Be direct about correctness, kind about style. *"This fits ComBat before the
split — that leaks"* is a good review comment. Approving a leaky pipeline
because a teammate is in a hurry costs everyone marks.

---

## 5. Releases

At each milestone (proposal demo, progress review, final submission):

```bash
git switch -c release/v1.0-proposal develop
# stabilise: docs, version bumps, no new features
# PR release/v1.0-proposal -> main, get approval, squash merge
git tag -a v1.0-proposal -m "Proposal presentation submission"
git push origin v1.0-proposal
# then PR main -> develop to carry back any release fixes
```

---

## 6. Repo setup — one-time, repo admin only

Branch protection is what makes the rule at the top of this file real. Until
this is configured, "don't push to main" is only a promise.

**Settings → Branches → Add branch ruleset** (or Branch protection rules), for
both `main` and `develop`:

- ✅ Require a pull request before merging
  - **Require approvals: `0`** ← this is what permits self-merge. GitHub
    disables the Approve button on your own PR, so any value ≥ 1 means you can
    never merge your own work.
  - ❌ Require review from Code Owners *(off — it would re-block self-merge on
    any path you own)*
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - Select: `Mobile Build`, `Desktop App Build`, `Backend Build`
  - **Only these three run today.** Lint, tests, and the data/notebook-output
    guards were deliberately dropped from CI for now — they'll come back as
    separate jobs (and get added to this list) once there's real app code and
    model code to check. Until then, review PRs for section 3/4 compliance
    by eye — see the "Reviewing" checklist below.
- ✅ Require conversation resolution before merging
- ✅ Block force pushes
- ✅ Restrict deletions
- ✅ **Do not allow bypassing the above settings** ← without this, admins
  (including the repo owner) can still push straight to `main`, and the rule
  is decorative.

With approvals at 0, **CI is the gate**. Keep the status checks required — they
are now the only automated thing standing between a bad commit and `develop`.

Also under **Settings → General → Pull Requests**: enable *Allow squash
merging* only, and *Automatically delete head branches*.

If you have the GitHub CLI authenticated, `scripts/setup-branch-protection.sh`
applies all of the above for both branches in one go:

```bash
./scripts/setup-branch-protection.sh              # self-merge allowed (default)
REVIEWERS=1 ./scripts/setup-branch-protection.sh  # require a second person
```

**CODEOWNERS is still useful with approvals at 0** — it auto-requests the right
reviewer so the owner of a module sees the PR, it just doesn't block the merge.
