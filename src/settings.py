from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIR.parent
load_dotenv(REPOSITORY_ROOT / ".env", override=False)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "stylme-voice")

CONTROL_PLANE_URL = os.getenv("LIVEKIT_CONTROL_PLANE_URL", "http://localhost:8081/v1")
INTERNAL_API_KEY = os.getenv("AI_INTERNAL_API_KEY") or os.getenv("CRON_SECRET", "")
DEFAULT_INBOUND_SWARM_ID = os.getenv(
    "LIVEKIT_DEFAULT_INBOUND_SWARM_ID", "swarm_default_inbound"
)
DEFAULT_OUTBOUND_SWARM_ID = os.getenv(
    "LIVEKIT_DEFAULT_OUTBOUND_SWARM_ID", "swarm_default_outbound"
)
PYTHON_API_BASE_URL = os.getenv("PYTHON_API_BASE_URL") or os.getenv(
    "NEXT_PUBLIC_API_BASE_URL", "http://localhost:8000/api/v1"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")


def validate_runtime_settings() -> None:
    required = {
        "LIVEKIT_API_KEY": LIVEKIT_API_KEY,
        "LIVEKIT_API_SECRET": LIVEKIT_API_SECRET,
        "AI_INTERNAL_API_KEY or CRON_SECRET": INTERNAL_API_KEY,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required root environment values: " + ", ".join(missing)
        )


def validate_provider_credentials(credentials: dict[str, str]) -> None:
    """Accept database-managed credentials first and environment fallbacks second."""
    effective = {
        "openai": credentials.get("openai") or OPENAI_API_KEY,
        "deepgram": credentials.get("deepgram") or DEEPGRAM_API_KEY,
        "sarvam": credentials.get("sarvam") or SARVAM_API_KEY,
    }
    missing = [provider for provider, value in effective.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing active voice provider credentials: " + ", ".join(missing)
        )
