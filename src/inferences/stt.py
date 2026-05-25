import os

from livekit.plugins import deepgram

from core.env import _env_bool, _env_int, _plugin_model
from settings import deepgram_api_key


def stt_model_name() -> str:
    return _plugin_model(os.getenv("DEEPGRAM_STT_MODEL", "nova-3"), "deepgram")


def _stt_provider_name() -> str:
    return "deepgram"


def build_stt(stt_language: str) -> deepgram.STT:
    language = (stt_language or "multi").strip() or "multi"

    # Deepgram Nova-3 with language="multi" auto-detects each spoken segment, so
    # callers can move between Hindi, English, and other supported languages
    # without the pipeline being pinned to one locale.
    #
    # interim_results stay on to power smart turn detection (the LiveKit multilingual
    # end-of-utterance model + Silero VAD) and the barge-in word gate.
    return deepgram.STT(
        model=stt_model_name(),
        language=language,
        api_key=deepgram_api_key,
        interim_results=True,
        punctuate=True,
        smart_format=_env_bool("DEEPGRAM_SMART_FORMAT", False),
        # Filler words ("um", "uh") improve end-of-turn detection accuracy.
        filler_words=_env_bool("DEEPGRAM_FILLER_WORDS", True),
        endpointing_ms=_env_int(
            "DEEPGRAM_ENDPOINTING_MS", 25, min_value=0, max_value=500
        ),
    )
