import os

from livekit.plugins import elevenlabs, sarvam

from core.env import _env_float, _env_int, _plugin_model
from inferences.voice import _use_elevenlabs_tts
from settings import (
    elevenlabs_api_key,
    elevenlabs_voice_id,
    sarvam_api_key,
    sarvam_model,
    sarvam_speaker,
    sarvam_target_language_code,
)


def _build_elevenlabs_tts(model: str) -> elevenlabs.TTS:
    # Minimal config — the plugin's own defaults are already the optimized path:
    # auto_mode streams a sentence at a time, the default mp3_22050_32 encoding has
    # the lowest time-to-first-byte, text normalization is "auto", and the
    # streaming WebSocket stays warm for 180s. No manual chunk schedule, encoding
    # override, or voice tuning needed.
    #
    # eleven_flash_v2_5 is multilingual, so the agent speaks Hindi by default and
    # follows the caller into any language. Override with ELEVENLABS_TTS_LANGUAGE.
    return elevenlabs.TTS(
        model=model,
        voice_id=elevenlabs_voice_id,
        api_key=elevenlabs_api_key,
        language=os.getenv("ELEVENLABS_TTS_LANGUAGE", "hi"),
        auto_mode=True,
    )


def _build_sarvam_tts(model: str) -> sarvam.TTS:
    return sarvam.TTS(
        model=model,
        target_language_code=sarvam_target_language_code,
        speaker=sarvam_speaker,
        api_key=sarvam_api_key,
        pace=_env_float("SARVAM_PACE", 1.0, min_value=0.5, max_value=2.0),
        temperature=_env_float(
            "SARVAM_TEMPERATURE", 0.6, min_value=0.01, max_value=1.0
        ),
        output_audio_bitrate=os.getenv("SARVAM_OUTPUT_AUDIO_BITRATE", "128k"),
        min_buffer_size=_env_int(
            "SARVAM_MIN_BUFFER_SIZE", 40, min_value=20, max_value=200
        ),
        max_chunk_length=_env_int(
            "SARVAM_MAX_CHUNK_LENGTH", 120, min_value=40, max_value=500
        ),
        speech_sample_rate=_env_int(
            "SARVAM_SPEECH_SAMPLE_RATE", 22050, min_value=8000, max_value=24000
        ),
    )


def _build_tts(model: str | None = None) -> elevenlabs.TTS | sarvam.TTS:
    if _use_elevenlabs_tts():
        selected_model = _plugin_model(
            model or os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
            "elevenlabs",
        )
        return _build_elevenlabs_tts(selected_model)

    selected_model = _plugin_model(model or sarvam_model, "sarvam")
    return _build_sarvam_tts(selected_model)
