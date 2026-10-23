#!/usr/bin/env bash
set -euo pipefail
unset GITHUB_TOKEN FORGEJO_TOKEN GITEA_TOKEN

# The workflow only supplies true for an exact OS/architecture/Node/lockfile key.
# npm ls also detects missing, invalid, and extraneous dependency installations.
for project in . prod; do
  if test "${MODULES_CACHE_HIT:-false}" = true \
    && test -d "$project/node_modules" \
    && npm --prefix "$project" ls --all --silent > /dev/null 2>&1; then
    echo "Reusing verified installed npm dependencies: $project"
  else
    echo "Installing npm dependencies from lockfile: $project"
    npm --prefix "$project" ci --prefer-offline --no-audit --no-fund
  fi
done
