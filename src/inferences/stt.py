from livekit.plugins import deepgram

from core.env import _env_bool, _env_int
from settings import deepgram_api_key


def build_stt(stt_model: str, stt_language: str) -> deepgram.STT:
    return deepgram.STT(
        model=stt_model,
        language=stt_language,
        api_key=deepgram_api_key,
        interim_results=True,
        no_delay=True,
        endpointing_ms=_env_int(
            "DEEPGRAM_ENDPOINTING_MS", 25, min_value=10, max_value=500
        ),
        # smart_format formats spoken numbers, emails, and phone numbers into their
        # written form (e.g. "nine one eight" -> "918"), which makes capturing
        # contact details far more reliable. Set DEEPGRAM_SMART_FORMAT=false to
        # trade this accuracy back for a little latency.
        smart_format=_env_bool("DEEPGRAM_SMART_FORMAT", True),
        filler_words=_env_bool("DEEPGRAM_FILLER_WORDS", False),
    )
