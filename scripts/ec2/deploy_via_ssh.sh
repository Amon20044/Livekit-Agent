#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <ec2_host> <ec2_user> <pem_file> [app_dir] [env_file] [ssh_port]"
  exit 1
fi

EC2_HOST="$1"
EC2_USER="$2"
PEM_FILE="$3"
APP_DIR="${4:-/opt/my-agent}"
ENV_FILE="${5:-/etc/my-agent.env}"
SSH_PORT="${6:-22}"

if [[ ! -f "$PEM_FILE" ]]; then
  echo "PEM file not found: $PEM_FILE"
  exit 1
fi

chmod 600 "$PEM_FILE"

tar \
  --exclude-vcs \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.ruff_cache' \
  --exclude='.pytest_cache' \
  -czf /tmp/my-agent.tar.gz .

scp -i "$PEM_FILE" -P "$SSH_PORT" /tmp/my-agent.tar.gz "$EC2_USER@$EC2_HOST:/tmp/my-agent.tar.gz"
scp -i "$PEM_FILE" -P "$SSH_PORT" scripts/ec2/start_agent_worker.sh "$EC2_USER@$EC2_HOST:/tmp/start_agent_worker.sh"

ssh -i "$PEM_FILE" -p "$SSH_PORT" "$EC2_USER@$EC2_HOST" "
  set -euo pipefail
  chmod +x /tmp/start_agent_worker.sh
  sudo mkdir -p '$APP_DIR'
  sudo tar -xzf /tmp/my-agent.tar.gz -C '$APP_DIR'
  APP_DIR='$APP_DIR' REPO_URL='' ENV_FILE='$ENV_FILE' /tmp/start_agent_worker.sh
"

echo "Deploy complete."
echo "Remote log: $APP_DIR/logs/agent.log"
