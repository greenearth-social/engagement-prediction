#!/usr/bin/env bash
set -euo pipefail

# Install this helper outside the checkout: updating main must not replace a
# shell script while Bash is still reading it. systemd sets WorkingDirectory.
if [[ "$(git branch --show-current)" != main ]]; then
    echo "Scheduled training requires the dedicated checkout to be on main." >&2
    exit 1
fi

# Ignored files such as .venv and config.automation.local.yml are allowed.
changes=$(git status --porcelain)
if [[ -n "$changes" ]]; then
    echo "Scheduled training checkout has local changes; refusing to update." >&2
    exit 1
fi

git fetch origin main
revision=$(git rev-parse FETCH_HEAD)
# A fast-forward merge alone also accepts a checkout ahead of the remote.
# Require the local revision to be an ancestor so every run uses remote main.
if ! git merge-base --is-ancestor HEAD "$revision"; then
    echo "Scheduled training checkout has local or divergent commits; refusing to update." >&2
    exit 1
fi

git merge --ff-only "$revision"
printf 'Scheduled training revision: %s\n' "$(git rev-parse HEAD)"
