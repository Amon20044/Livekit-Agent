#!/usr/bin/env bash
set -euo pipefail

# Encrypts .env.local -> .env.encoded, stages all changes,
# uses Claude Code to generate a commit message, commits and pushes.
#
# Usage: ./scripts/push-env.sh [path-to-env-file]

ENV_FILE="${1:-.env.local}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found"
  exit 1
fi

# Read ENV_LOCAL_B64 from the env file itself
ENV_LOCAL_B64=$(grep -m1 '^ENV_LOCAL_B64=' "$ENV_FILE" | cut -d'=' -f2-)
if [[ -z "$ENV_LOCAL_B64" ]]; then
  echo "Error: ENV_LOCAL_B64 not found in $ENV_FILE"
  exit 1
fi
export ENV_LOCAL_B64
openssl enc -aes-256-cbc -pbkdf2 -a -in "$ENV_FILE" -out .env.encoded -pass env:ENV_LOCAL_B64
echo "Encrypted $ENV_FILE -> .env.encoded"

# Stage all changes
git add -A

CHANGED=$(git diff --cached --name-only)
if [ -z "$CHANGED" ]; then
  echo "Nothing to commit."
  exit 0
fi

# Use Claude Code to generate commit message
echo "Generating commit message..."
MSG=$(claude -p "Look at this git diff and generate a single-line commit message (max 72 chars). No quotes, no prefixes like 'feat:' or 'fix:' unless appropriate. Just describe what changed concisely.

$(git diff --cached --stat)

$(git diff --cached)")

git commit -m "$MSG"
git push
echo "Pushed: $MSG"
