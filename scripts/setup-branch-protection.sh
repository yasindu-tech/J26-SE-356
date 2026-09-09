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

set -euo pipefail

REPO="${REPO:-yasindu-tech/J26-SE-356}"
REVIEWERS="${REVIEWERS:-1}"

command -v gh >/dev/null 2>&1 || { echo "error: gh CLI not found — https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: not authenticated — run 'gh auth login'"; exit 1; }

echo "Repo:            $REPO"
echo "Required reviews: $REVIEWERS"
echo

# Status check names must match the `name:` of each job in .github/workflows/ci.yml
CHECKS='["lint-python","test-python","lint-js","build-js","guard-no-data","guard-notebook-outputs"]'

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
    "require_code_owner_reviews": true,
    "require_last_push_approval": true
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
echo "Note: 'enforce_admins: true' means the rules apply to the repo owner too."
echo "That is deliberate — see CONTRIBUTING.md section 6."
