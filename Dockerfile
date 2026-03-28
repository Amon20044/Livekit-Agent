# syntax=docker/dockerfile:1

# =========================================================
#  LiveKit AI Agent - Production Dockerfile
#  Matches the proven legacy agent-worker setup
# =========================================================

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV XDG_CACHE_HOME=/app/.cache
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies (same set as the working legacy build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libportaudio2 \
    portaudio19-dev \
    alsa-utils \
    ffmpeg \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./
RUN mkdir -p src

# Install Python dependencies
RUN uv sync --locked

# Copy application code
COPY . .

# Pre-download ML models (VAD, turn detector)
RUN uv run src/agent.py download-files

# Start the agent
CMD ["uv", "run", "src/agent.py", "start"]
