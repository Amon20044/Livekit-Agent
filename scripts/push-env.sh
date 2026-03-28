#!/usr/bin/env bash
set -euo pipefail

# Encodes .env.local and uploads it as a GitHub secret, then stages, commits,
# and pushes code changes. The CI workflow decodes the secret on the server.
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
git add .env.encoded
echo "Encrypted $ENV_FILE to .env.encoded."

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
