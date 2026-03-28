#!/usr/bin/env bash
set -euo pipefail

# Encodes .env.local, commits .env.encoded, and pushes to GitHub.
# The deploy workflow decodes it on EC2 and deletes it from the repo.
#
# Usage: ./scripts/push-env.sh

ENV_FILE="${1:-.env.local}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found"
  exit 1
fi

base64 -w 0 "$ENV_FILE" > .env.encoded 2>/dev/null || base64 -i "$ENV_FILE" | tr -d '\n' > .env.encoded
git add .env.encoded
git commit -m "chore: update encoded env"
git push
echo "Pushed. Workflow will decode on EC2 and remove .env.encoded from repo."
