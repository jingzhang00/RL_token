#!/usr/bin/env bash

set -euo pipefail

commit_message="${*:-ver3: fix online rollout reply buffer}"

git rev-parse --is-inside-work-tree >/dev/null
git add -A

if git diff --cached --quiet; then
    echo "no changes to commit"
else
    git commit -m "$commit_message"
fi

git push