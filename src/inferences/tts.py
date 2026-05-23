import logging
import os

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

logger = logging.getLogger("agent")

# Output formats the ElevenLabs plugin accepts. PCM is preferred for voice agents:
# it skips the MP3 decode stage, so a late or partial frame on the streaming
# WebSocket degrades to a tiny silence instead of stalling the decoder (which is
# what surfaces as "lost packets"/choppy audio mid-call). The plugin default is the
# low-bitrate "mp3_22050_32"; we default to 24 kHz PCM for clean, robust audio.
_ELEVENLABS_ENCODINGS = frozenset(
    {
        "mp3_22050_32",
        "mp3_24000_48",
        "mp3_44100",
        "mp3_44100_32",
        "mp3_44100_64",
        "mp3_44100_96",
        "mp3_44100_128",
        "mp3_44100_192",
        "opus_48000_32",
        "opus_48000_64",
        "opus_48000_96",
        "opus_48000_128",
        "opus_48000_192",
        "pcm_8000",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_32000",
        "pcm_44100",
        "pcm_48000",
    }
)
_DEFAULT_ELEVENLABS_ENCODING = "pcm_24000"
_ELEVENLABS_TEXT_NORMALIZATION = frozenset({"auto", "on", "off"})


def _elevenlabs_encoding() -> str:
    value = (
        (os.getenv("ELEVENLABS_OUTPUT_FORMAT") or _DEFAULT_ELEVENLABS_ENCODING)
        .strip()
        .lower()
    )
    if value in _ELEVENLABS_ENCODINGS:
        return value
    logger.warning(
        "Invalid ELEVENLABS_OUTPUT_FORMAT=%r; using %s",
        value,
        _DEFAULT_ELEVENLABS_ENCODING,
    )
    return _DEFAULT_ELEVENLABS_ENCODING


def _elevenlabs_text_normalization() -> str:
    value = (os.getenv("ELEVENLABS_TEXT_NORMALIZATION") or "auto").strip().lower()
    if value in _ELEVENLABS_TEXT_NORMALIZATION:
        return value
    logger.warning("Invalid ELEVENLABS_TEXT_NORMALIZATION=%r; using auto", value)
    return "auto"


def _build_elevenlabs_tts(model: str) -> elevenlabs.TTS:
    return elevenlabs.TTS(
        model=model,
        voice_id=elevenlabs_voice_id,
        api_key=elevenlabs_api_key,
        language=os.getenv("ELEVENLABS_TTS_LANGUAGE", "en"),
        # PCM by default: no decode stage, so jittery/partial WS frames don't stall
        # playback the way a corrupt MP3 frame does.
        encoding=_elevenlabs_encoding(),
        streaming_latency=_env_int(
            "ELEVENLABS_STREAMING_LATENCY", 3, min_value=0, max_value=4
        ),
        # Keep the streaming WebSocket warm across conversational pauses so a quiet
        # stretch doesn't drop the connection and force a reconnect (heard as a gap
        # or lost first syllable on the next reply). 180s is the ElevenLabs maximum.
        inactivity_timeout=_env_int(
            "ELEVENLABS_INACTIVITY_TIMEOUT", 180, min_value=20, max_value=180
        ),
        # "auto" lets ElevenLabs normalize numbers/emails for natural readback while
        # staying fast; set to "off" for lowest latency or "on" to force it.
        apply_text_normalization=_elevenlabs_text_normalization(),
        auto_mode=_env_bool("ELEVENLABS_AUTO_MODE", True),
        chunk_length_schedule=[
            _env_int("ELEVENLABS_CHUNK_1", 50, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_2", 70, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_3", 100, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_4", 140, min_value=50, max_value=500),
        ],
        voice_settings=elevenlabs.VoiceSettings(
            stability=_env_float(
                "ELEVENLABS_STABILITY", 0.45, min_value=0.0, max_value=1.0
            ),
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
