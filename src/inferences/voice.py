import os

from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from core.env import _env_bool


def _use_elevenlabs_tts() -> bool:
    return _env_bool("USE_EL", True)


def _tts_provider(use_elevenlabs: bool | None = None) -> str:
    if use_elevenlabs is None:
        use_elevenlabs = _use_elevenlabs_tts()
    return "elevenlabs" if use_elevenlabs else "sarvam"


def _stt_language(use_elevenlabs: bool | None = None) -> str:
    if _tts_provider(use_elevenlabs) == "elevenlabs":
        return os.getenv(
            "ELEVENLABS_SPEECHMATICS_STT_LANGUAGE",
            os.getenv("SPEECHMATICS_STT_LANGUAGE", "en"),
        )
    return os.getenv("SPEECHMATICS_STT_LANGUAGE", "en")


def _deepgram_language(use_elevenlabs: bool | None = None) -> str:
    return _stt_language(use_elevenlabs)


def _build_turn_detector() -> EnglishModel | MultilingualModel:
    if _use_elevenlabs_tts():
        return EnglishModel()
    return MultilingualModel()
