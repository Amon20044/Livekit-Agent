#!/usr/bin/env bash
set -euo pipefail

# Encodes .env.local -> .env.encoded, stages everything, commits with a
# summary of changes, and pushes.
# Usage: ./scripts/push-env.sh [path-to-env-file]

ENV_FILE="${1:-.env.local}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found"
  exit 1
fi

# Encode env
base64 -w 0 "$ENV_FILE" > .env.encoded 2>/dev/null || base64 -i "$ENV_FILE" | tr -d '\n' > .env.encoded

# Stage all changes
git add -A

# Build commit message from changed files
CHANGED=$(git diff --cached --name-only | head -20)
if [ -z "$CHANGED" ]; then
  echo "Nothing to commit."
  exit 0
fi

MSG=$(echo "$CHANGED" | sed 's|src/||;s|\.py$||;s|\..*$||' | sort -u | paste -sd ', ' -)
git commit -m "deploy: $MSG"
git push

echo "Pushed."
