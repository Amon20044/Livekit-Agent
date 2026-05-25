import os
from typing import Any

from livekit.plugins import speechmatics

from core.env import _env_bool, _env_float, _env_int


def _speechmatics_operating_point() -> speechmatics.OperatingPoint:
    value = os.getenv("SPEECHMATICS_OPERATING_POINT", "enhanced")
    try:
        return speechmatics.OperatingPoint(value.strip().lower())
    except ValueError:
        return speechmatics.OperatingPoint.ENHANCED


def speechmatics_operating_point_name() -> str:
    return _speechmatics_operating_point().value


def _additional_vocab() -> list[speechmatics.AdditionalVocabEntry]:
    return [
        speechmatics.AdditionalVocabEntry(
            content="LiveKit",
            sounds_like=["live kit"],
        ),
        speechmatics.AdditionalVocabEntry(
            content="Woice",
            sounds_like=["voice"],
        ),
        speechmatics.AdditionalVocabEntry(
            content="2000",
            sounds_like=["two thousands"],
        ),
        speechmatics.AdditionalVocabEntry(
            content="Speechmatics",
            sounds_like=["speech-matics"],
        ),
    ]


def build_stt(stt_language: str) -> speechmatics.STT:
    kwargs: dict[str, Any] = {
        "language": stt_language,
        "operating_point": _speechmatics_operating_point(),
        "include_partials": _env_bool("SPEECHMATICS_INCLUDE_PARTIALS", True),
        "max_delay": _env_float(
            "SPEECHMATICS_MAX_DELAY", 0.7, min_value=0.7, max_value=4.0
        ),
        "end_of_utterance_silence_trigger": _env_float(
            "SPEECHMATICS_END_OF_UTTERANCE_SILENCE_TRIGGER",
            0.5,
            min_value=0.05,
            max_value=1.95,
        ),
        "turn_detection_mode": speechmatics.TurnDetectionMode.SMART_TURN,
        "enable_diarization": _env_bool("SPEECHMATICS_ENABLE_DIARIZATION", True),
        "speaker_active_format": os.getenv(
            "SPEECHMATICS_SPEAKER_ACTIVE_FORMAT",
            "<{speaker_id}>{text}</{speaker_id}>",
        ),
        "speaker_passive_format": os.getenv(
            "SPEECHMATICS_SPEAKER_PASSIVE_FORMAT",
            "[{speaker_id}^PASSIVE*] {text}",
        ),
        # Lock onto the customer: group words close together to the same speaker so
        # a brief background voice doesn't get split out and transcribed as the
        # caller. Reduces background-speaker bleed when diarization is on.
        "prefer_current_speaker": _env_bool(
            "SPEECHMATICS_PREFER_CURRENT_SPEAKER", True
        ),
        "additional_vocab": _additional_vocab(),
    }

    # Optional diarization tuning. Lower sensitivity merges nearby voices into the
    # main speaker (good for noisy rooms); max_speakers caps how many the engine
    # splits out. Only sent when explicitly configured so plugin defaults stand.
    sensitivity = os.getenv("SPEECHMATICS_SPEAKER_SENSITIVITY")
    if sensitivity is not None:
        kwargs["speaker_sensitivity"] = _env_float(
            "SPEECHMATICS_SPEAKER_SENSITIVITY", 0.5, min_value=0.01, max_value=0.99
        )
    if os.getenv("SPEECHMATICS_MAX_SPEAKERS") is not None:
        kwargs["max_speakers"] = _env_int(
            "SPEECHMATICS_MAX_SPEAKERS", 2, min_value=2, max_value=100
        )

    return speechmatics.STT(**kwargs)
