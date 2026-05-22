import os

from livekit.plugins import elevenlabs, sarvam

from core.env import _env_bool, _env_float, _env_int, _plugin_model
from inferences.voice import _use_elevenlabs_tts
from settings import (
    elevenlabs_api_key,
    elevenlabs_voice_id,
    sarvam_api_key,
    sarvam_speaker,
    sarvam_target_language_code,
    sarvam_model,
)


def _build_elevenlabs_tts(model: str) -> elevenlabs.TTS:
    return elevenlabs.TTS(
        model=model,
        voice_id=elevenlabs_voice_id,
        api_key=elevenlabs_api_key,
        language=os.getenv("ELEVENLABS_TTS_LANGUAGE", "en"),
        streaming_latency=_env_int("ELEVENLABS_STREAMING_LATENCY", 3, min_value=0, max_value=4),
        auto_mode=_env_bool("ELEVENLABS_AUTO_MODE", True),
        chunk_length_schedule=[
            _env_int("ELEVENLABS_CHUNK_1", 50, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_2", 70, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_3", 100, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_4", 140, min_value=50, max_value=500),
        ],
        voice_settings=elevenlabs.VoiceSettings(
            stability=_env_float("ELEVENLABS_STABILITY", 0.45, min_value=0.0, max_value=1.0),
            similarity_boost=_env_float(
                "ELEVENLABS_SIMILARITY_BOOST", 0.75, min_value=0.0, max_value=1.0
            ),
            speed=_env_float("ELEVENLABS_SPEED", 1.08, min_value=0.25, max_value=4.0),
            use_speaker_boost=_env_bool("ELEVENLABS_SPEAKER_BOOST", False),
        ),
        sync_alignment=_env_bool("ELEVENLABS_SYNC_ALIGNMENT", False),
    )


def _build_sarvam_tts(model: str) -> sarvam.TTS:
    return sarvam.TTS(
        model=model,
        target_language_code=sarvam_target_language_code,
        speaker=sarvam_speaker,
        api_key=sarvam_api_key,
        pace=_env_float("SARVAM_PACE", 1.0, min_value=0.5, max_value=2.0),
        temperature=_env_float("SARVAM_TEMPERATURE", 0.6, min_value=0.01, max_value=1.0),
        output_audio_bitrate=os.getenv("SARVAM_OUTPUT_AUDIO_BITRATE", "128k"),
        min_buffer_size=_env_int("SARVAM_MIN_BUFFER_SIZE", 40, min_value=20, max_value=200),
        max_chunk_length=_env_int("SARVAM_MAX_CHUNK_LENGTH", 120, min_value=40, max_value=500),
        speech_sample_rate=_env_int("SARVAM_SPEECH_SAMPLE_RATE", 22050, min_value=8000, max_value=24000),
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
