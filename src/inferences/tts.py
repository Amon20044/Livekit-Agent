from livekit.plugins import sarvam

from app.models import VoiceConfig
from settings import SARVAM_API_KEY


def build_tts(config: VoiceConfig, api_key: str = "") -> sarvam.TTS:
    target_language = "hi-IN" if config.language == "multi" else config.language
    return sarvam.TTS(
        model=config.tts_model,
        target_language_code=target_language,
        speaker=config.speaker,
        api_key=api_key or SARVAM_API_KEY,
        pace=config.pace,
    )
