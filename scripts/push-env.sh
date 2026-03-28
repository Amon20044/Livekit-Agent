#!/usr/bin/env bash
set -euo pipefail

# Encodes .env.local -> base64 -> writes to .env.encoded in the repo root.
# This file gets pushed with the code. The deploy workflow decodes it on EC2.
#
# Usage: ./scripts/push-env.sh

ENV_FILE="${1:-.env.local}"
ENCODED_FILE=".env.encoded"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found"
  exit 1
fi

echo "Encoding $ENV_FILE -> $ENCODED_FILE"
base64 -w 0 "$ENV_FILE" > "$ENCODED_FILE" 2>/dev/null || base64 -i "$ENV_FILE" | tr -d '\n' > "$ENCODED_FILE"
echo "Done. Now commit and push $ENCODED_FILE with your code."
