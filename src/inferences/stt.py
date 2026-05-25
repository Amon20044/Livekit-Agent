import os

from livekit.plugins import deepgram

from core.env import _env_bool, _env_int, _plugin_model
from settings import deepgram_api_key


def stt_model_name() -> str:
    return _plugin_model(os.getenv("DEEPGRAM_STT_MODEL", "nova-3"), "deepgram")


def build_stt(stt_language: str) -> deepgram.STT:
    # Deepgram nova-3 with language="multi" auto-detects the spoken language and
    # code-switches in real time, so the caller can speak any supported language and
    # the agent understands it without being pinned to one language.
    #
    # interim_results stay on to power smart turn detection (the LiveKit multilingual
    # end-of-utterance model + Silero VAD) and the barge-in word gate. They never
    # reach the LLM: with preemptive generation off, the LLM is called exactly once
    # per turn, on the final transcript.
    return deepgram.STT(
        model=stt_model_name(),
        language=stt_language,
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
