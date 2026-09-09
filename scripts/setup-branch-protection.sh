#!/usr/bin/env bash
#
# Applies branch protection to `main` and `develop`.
#
# This is the script that turns "please don't push to main" into something the
# server enforces. Run it once, from a machine with the GitHub CLI authenticated
# as a user with **admin** rights on the repo.
#
#   brew install gh          # or: https://cli.github.com
#   gh auth login
#   ./scripts/setup-branch-protection.sh
#
# Re-running is safe — it overwrites the rules with the same values.
#
# ---------------------------------------------------------------------------
# SELF-MERGE IS ALLOWED (deliberate choice for a 4-person team)
#
# REVIEWERS defaults to 0. This means you can merge your own PR once CI is
# green, without waiting for a teammate.
#
# Why 0 and not 1: **GitHub does not let you approve your own pull request.**
# The Approve button is disabled on PRs you authored. So any value >= 1 makes
# self-merge impossible no matter what else is configured — there is no
# "allow self-approval" switch to turn on.
#
# What still protects the branch with REVIEWERS=0:
#   - a PR is still REQUIRED (no direct pushes to main/develop, for anyone)
#   - CI must pass before the merge button turns green
#   - conversations must be resolved
#   - no force pushes, no branch deletion
#   - enforce_admins=true, so the above applies to the repo owner too
#
# If you later want true peer review, set REVIEWERS=1 and accept that every PR
# then needs a second person:
#   REVIEWERS=1 ./scripts/setup-branch-protection.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${REPO:-yasindu-tech/J26-SE-356}"
REVIEWERS="${REVIEWERS:-0}"

command -v gh >/dev/null 2>&1 || { echo "error: gh CLI not found — https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: not authenticated — run 'gh auth login'"; exit 1; }

echo "Repo:             $REPO"
echo "Required reviews: $REVIEWERS  $([ "$REVIEWERS" -eq 0 ] && echo '(self-merge allowed — CI still gates)' || echo '(a second person must approve)')"
echo

# Status check names must match the `name:` of each job in .github/workflows/ci.yml
# Only the three build jobs are required today. Lint, tests and the
# data/notebook guards were removed from CI for now — add their check names
# back here when those jobs come back (see CONTRIBUTING.md §6).
CHECKS='["Mobile Build","Desktop App Build","Backend Build"]'

protect () {
  local BRANCH="$1"
  echo "→ Protecting '$BRANCH'..."

  # Ensure the branch exists before trying to protect it
  if ! gh api "repos/$REPO/branches/$BRANCH" >/dev/null 2>&1; then
    echo "  ! branch '$BRANCH' does not exist yet — create and push it first, skipping."
    return
  fi

  gh api -X PUT "repos/$REPO/branches/$BRANCH/protection" \
    -H "Accept: application/vnd.github+json" \
    --input - <<JSON >/dev/null
{
  "required_status_checks": {
    "strict": true,
    "contexts": $CHECKS
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": $REVIEWERS,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON
  echo "  ✓ $BRANCH protected"
}

protect main
protect develop

echo
echo "→ Repo merge settings..."
gh api -X PATCH "repos/$REPO" \
  -f allow_squash_merge=true \
  -F allow_merge_commit=false \
  -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true \
  -F allow_auto_merge=true >/dev/null
echo "  ✓ squash-only merges, auto-delete merged branches"

echo
echo "Done. Verify at: https://github.com/$REPO/settings/branches"
echo
echo "What this means in practice:"
echo "  • You CANNOT push directly to main or develop — not even as the owner"
echo "    ('enforce_admins: true' is deliberate)."
if [ "$REVIEWERS" -eq 0 ]; then
echo "  • You CAN merge your own PR, once CI is green and threads are resolved."
fi
echo "  • Every change goes through a PR into develop. See CONTRIBUTING.md."
