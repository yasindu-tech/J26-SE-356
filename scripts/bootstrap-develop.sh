#!/usr/bin/env bash
#
# Creates the `develop` branch off `main` and pushes it, then points the repo's
# default branch at `develop` so new PRs target it automatically.
#
# Run BEFORE setup-branch-protection.sh (you cannot protect a branch that does
# not exist yet).

set -euo pipefail
REPO="${REPO:-yasindu-tech/J26-SE-356}"

git switch main
git pull origin main

if git show-ref --verify --quiet refs/heads/develop; then
  echo "local 'develop' already exists"
else
  git switch -c develop
fi

git push -u origin develop

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh api -X PATCH "repos/$REPO" -f default_branch=develop >/dev/null
  echo "✓ default branch set to 'develop'"
else
  echo "! gh not available — set the default branch to 'develop' manually:"
  echo "  Settings → General → Default branch"
fi
