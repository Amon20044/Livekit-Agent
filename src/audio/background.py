import logging
import os

from livekit.agents import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip

from core.env import _env_bool, _env_float, _env_int

logger = logging.getLogger("agent")


def _builtin_audio_clip(name: str, default: BuiltinAudioClip) -> BuiltinAudioClip:
    normalized = name.strip().upper()
    if normalized in BuiltinAudioClip.__members__:
        return BuiltinAudioClip[normalized]

    logger.warning("Unknown builtin audio clip %r; using %s", name, default.name)
    return default


def _build_background_audio_player() -> BackgroundAudioPlayer | None:
    if not _env_bool("BACKGROUND_AUDIO_ENABLED", True):
        return None

    ambient_clip = _builtin_audio_clip(
        os.getenv("BACKGROUND_AMBIENT_CLIP", "OFFICE_AMBIENCE"),
        BuiltinAudioClip.OFFICE_AMBIENCE,
    )
    ambient_sound = None
    if _env_bool("BACKGROUND_AMBIENT_SOUND_ENABLED", False):
        ambient_sound = AudioConfig(
            ambient_clip,
            volume=_env_float(
                "BACKGROUND_AMBIENT_VOLUME", 0.18, min_value=0.0, max_value=1.0
            ),
        )

    thinking_sound = None
    if _env_bool("BACKGROUND_THINKING_SOUND_ENABLED", True):
        thinking_clip = _builtin_audio_clip(
            os.getenv("BACKGROUND_THINKING_CLIP", "HOLD_MUSIC"),
            BuiltinAudioClip.HOLD_MUSIC,
        )
        thinking_sound = AudioConfig(
            thinking_clip,
            volume=_env_float(
                "BACKGROUND_THINKING_VOLUME", 0.10, min_value=0.0, max_value=1.0
            ),
        )
        if thinking_clip == BuiltinAudioClip.KEYBOARD_TYPING:
            thinking_sound = [
                AudioConfig(
                    thinking_clip,
                    volume=_env_float(
                        "BACKGROUND_THINKING_VOLUME",
                        0.10,
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    probability=0.75,
                ),
                AudioConfig(
                    BuiltinAudioClip.KEYBOARD_TYPING2,
                    volume=_env_float(
                        "BACKGROUND_THINKING_VOLUME_ALT",
                        0.08,
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    probability=0.25,
                ),
            ]

    return BackgroundAudioPlayer(
        ambient_sound=ambient_sound,
        thinking_sound=thinking_sound,
        stream_timeout_ms=_env_int(
            "BACKGROUND_AUDIO_STREAM_TIMEOUT_MS", 200, min_value=50, max_value=5000
        ),
    )
