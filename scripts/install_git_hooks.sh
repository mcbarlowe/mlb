#!/usr/bin/env bash
# Point this repository's hooks at the tracked scripts/hooks directory.
#
# core.hooksPath is used instead of copying into .git/hooks so the hooks stay
# version controlled and an update to one takes effect without reinstalling.
# Note that it REPLACES .git/hooks wholesale: any hook living only there stops
# running, which is why this prints what it found before switching.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

existing="$(git rev-parse --git-path hooks)"
local_hooks="$(find "$existing" -maxdepth 1 -type f ! -name '*.sample' ! -name '*.bak' -exec basename {} \; 2>/dev/null | sort)"
if [ -n "$local_hooks" ]; then
    echo "note: these hooks live only in $existing and will stop running:"
    echo "$local_hooks" | sed 's/^/  /'
fi

chmod +x scripts/hooks/*
git config core.hooksPath scripts/hooks

echo "core.hooksPath = $(git config --get core.hooksPath)"
printf 'active hooks:'
for hook in scripts/hooks/*; do
    printf ' %s' "$(basename "$hook")"
done
echo
echo
echo "post-commit mirrors docs into Notion in the background when a mirrored"
echo "file changes; output goes to data/notion/sync.log."
echo "Disable with: git config --unset core.hooksPath"
