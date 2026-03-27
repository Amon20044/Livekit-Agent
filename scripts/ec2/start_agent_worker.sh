#!/usr/bin/env bash
set -euo pipefail

# Minimal bootstrap/start script for a LiveKit agent worker on Ubuntu EC2.
# Environment variables accepted:
# - APP_DIR (default: /opt/my-agent)
# - BRANCH (default: main)
# - REPO_URL (optional; used if APP_DIR is not a git checkout)
# - ENV_FILE (default: /etc/my-agent.env)
APP_DIR="${APP_DIR:-/opt/my-agent}"
BRANCH="${BRANCH:-main}"
REPO_URL="${REPO_URL:-}"
ENV_FILE="${ENV_FILE:-/etc/my-agent.env}"

if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    ${SUDO} apt-get update -y
    if ! command -v git >/dev/null 2>&1; then
      ${SUDO} apt-get install -y git ca-certificates
    fi
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
      ${SUDO} apt-get install -y wget
    fi
  elif command -v dnf >/dev/null 2>&1; then
    if ! command -v git >/dev/null 2>&1; then
      ${SUDO} dnf install -y git ca-certificates
    fi
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
      ${SUDO} dnf install -y wget
    fi
  elif command -v yum >/dev/null 2>&1; then
    if ! command -v git >/dev/null 2>&1; then
      ${SUDO} yum install -y git ca-certificates
    fi
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
      ${SUDO} yum install -y wget
    fi
  else
    echo "No supported package manager found (apt-get/dnf/yum)."
    exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither curl nor wget is available to install uv."
    exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  # Common path when script runs in non-login shells.
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation failed or uv is not in PATH."
  exit 1
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  if [[ -z "$REPO_URL" ]]; then
    if [[ ! -f "$APP_DIR/src/agent.py" ]]; then
      echo "APP_DIR is not a git checkout and no agent source was found."
      echo "Set REPO_URL or copy the repository contents into $APP_DIR."
      exit 1
    fi
  else
    ${SUDO} mkdir -p "$(dirname "$APP_DIR")"
    ${SUDO} rm -rf "$APP_DIR"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi
fi

if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$BRANCH"
fi

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  set -a
  . "$ENV_FILE"
  set +a
else
  echo "Warning: env file not found at $ENV_FILE"
fi

cd "$APP_DIR"
mkdir -p logs

if [[ -f "uv.lock" ]]; then
  uv sync --locked
else
  echo "Warning: uv.lock not found; running 'uv sync' without --locked"
  uv sync
fi
uv run python src/agent.py download-files

pkill -f "src/agent.py start" || true
nohup uv run python src/agent.py start >> "$APP_DIR/logs/agent.log" 2>&1 &

echo "Agent worker started."
echo "Logs: $APP_DIR/logs/agent.log"
