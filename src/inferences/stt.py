from livekit.plugins import deepgram

from app.models import VoiceConfig
from settings import DEEPGRAM_API_KEY

_DEEPGRAM_NOVA_3_BASE_LANGUAGES = {"bn", "en", "hi", "kn", "mr", "ta", "te"}


def deepgram_language(language: str) -> str:
    """Translate Sarvam's Indian locales to codes accepted by Deepgram Nova-3."""
    configured = (language or "multi").strip()
    if configured in {"multi", "en-IN"}:
        return configured

    base_language = configured.partition("-")[0].lower()
    if base_language in _DEEPGRAM_NOVA_3_BASE_LANGUAGES:
        return base_language
    return "multi"


def build_stt(config: VoiceConfig, api_key: str = "") -> deepgram.STT:
    return deepgram.STT(
        model=config.stt_model,
        language=deepgram_language(config.language),
        api_key=api_key or DEEPGRAM_API_KEY,
        interim_results=True,
        punctuate=True,
        filler_words=True,
        smart_format=True,
        endpointing_ms=25,
    )
