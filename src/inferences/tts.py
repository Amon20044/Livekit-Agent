import os
from typing import Literal

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins import elevenlabs, sarvam

from core.env import _env_bool, _env_float, _env_int, _plugin_model
from inferences.voice import _use_elevenlabs_tts
from settings import (
    elevenlabs_api_key,
    elevenlabs_voice_id,
    sarvam_api_key,
    sarvam_model,
    sarvam_speaker,
    sarvam_target_language_code,
)

_ELEVENLABS_TEXT_NORMALIZATION = {"auto", "off", "on"}


def _elevenlabs_language() -> NotGivenOr[str]:
    value = (os.getenv("ELEVENLABS_TTS_LANGUAGE", "auto") or "").strip()
    if not value or value.lower() == "auto":
        return NOT_GIVEN
    return value


def _elevenlabs_text_normalization() -> Literal["auto", "off", "on"]:
    value = (os.getenv("ELEVENLABS_TEXT_NORMALIZATION", "auto") or "").strip().lower()
    if value in _ELEVENLABS_TEXT_NORMALIZATION:
        return value  # type: ignore[return-value]
    return "auto"


def _build_elevenlabs_tts(model: str) -> elevenlabs.TTS:
    # The plugin's optimized streaming path is auto_mode with its sentence
    # tokenizer: it flushes each sentence/phrase without a manual chunk schedule.
    # mp3_22050_32 is also the plugin default because it has the lowest measured
    # time-to-first-byte in the bundled ElevenLabs plugin.
    #
    # ELEVENLABS_TTS_LANGUAGE defaults to "auto" here, so ElevenLabs infers the
    # language from the generated text. Pin it only for intentionally
    # single-language deployments.
    return elevenlabs.TTS(
        model=model,
        voice_id=elevenlabs_voice_id,
        api_key=elevenlabs_api_key,
        language=_elevenlabs_language(),
        auto_mode=True,
        inactivity_timeout=_env_int(
            "ELEVENLABS_INACTIVITY_TIMEOUT", 180, min_value=20, max_value=180
        ),
        apply_text_normalization=_elevenlabs_text_normalization(),
        sync_alignment=_env_bool("ELEVENLABS_SYNC_ALIGNMENT", True),
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
