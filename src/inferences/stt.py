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
        endpointing_ms=_env_int("DEEPGRAM_ENDPOINTING_MS", 25, min_value=10, max_value=500),
        smart_format=_env_bool("DEEPGRAM_SMART_FORMAT", False),
        filler_words=_env_bool("DEEPGRAM_FILLER_WORDS", False),
    )
