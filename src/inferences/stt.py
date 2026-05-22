import os

from livekit.plugins import speechmatics

from core.env import _env_bool, _env_float


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
    return speechmatics.STT(
        language=stt_language,
        operating_point=_speechmatics_operating_point(),
        include_partials=_env_bool("SPEECHMATICS_INCLUDE_PARTIALS", True),
        max_delay=_env_float(
            "SPEECHMATICS_MAX_DELAY", 0.7, min_value=0.7, max_value=4.0
        ),
        end_of_utterance_silence_trigger=_env_float(
            "SPEECHMATICS_END_OF_UTTERANCE_SILENCE_TRIGGER",
            0.5,
            min_value=0.05,
            max_value=1.95,
        ),
        turn_detection_mode=speechmatics.TurnDetectionMode.SMART_TURN,
        enable_diarization=_env_bool("SPEECHMATICS_ENABLE_DIARIZATION", True),
        speaker_active_format=os.getenv(
            "SPEECHMATICS_SPEAKER_ACTIVE_FORMAT",
            "<{speaker_id}>{text}</{speaker_id}>",
        ),
        speaker_passive_format=os.getenv(
            "SPEECHMATICS_SPEAKER_PASSIVE_FORMAT",
            "[{speaker_id}^PASSIVE*] {text}",
        ),
        additional_vocab=_additional_vocab(),
    )
