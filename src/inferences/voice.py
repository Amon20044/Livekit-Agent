import os

from livekit.plugins.turn_detector.multilingual import MultilingualModel

from core.env import _env_bool


def _use_elevenlabs_tts() -> bool:
    return _env_bool("USE_EL", True)


def _tts_provider(use_elevenlabs: bool | None = None) -> str:
    if use_elevenlabs is None:
        use_elevenlabs = _use_elevenlabs_tts()
    return "elevenlabs" if use_elevenlabs else "sarvam"


def _stt_language(use_elevenlabs: bool | None = None) -> str:
    # Hindi by default for both TTS providers; the multilingual STT still handles
    # English and code-switching. Override with SPEECHMATICS_STT_LANGUAGE.
    return os.getenv("SPEECHMATICS_STT_LANGUAGE", "hi")


def _deepgram_language(use_elevenlabs: bool | None = None) -> str:
    return _stt_language(use_elevenlabs)


def _build_turn_detector() -> MultilingualModel:
    # Multilingual turn detection for every voice so Hindi (and any language the
    # caller switches to) gets correct end-of-turn handling.
    return MultilingualModel()
