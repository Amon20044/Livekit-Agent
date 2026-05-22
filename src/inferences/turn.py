from typing import Any

import os

from livekit.agents import TurnHandlingOptions

from core.env import _env_bool, _env_float, _env_int
from inferences.voice import _build_turn_detector

_DEFAULT_TURN_DETECTION = object()


def _build_turn_handling_options(
    turn_detection: Any = _DEFAULT_TURN_DETECTION,
) -> TurnHandlingOptions:
    if turn_detection is _DEFAULT_TURN_DETECTION:
        turn_detection = _build_turn_detector()

    return TurnHandlingOptions(
        turn_detection=turn_detection,
        endpointing={
            "mode": os.getenv("ENDPOINTING_MODE", "dynamic"),
            "min_delay": _env_float(
                "MIN_ENDPOINTING_DELAY", 0.22, min_value=0.05, max_value=2.0
            ),
            "max_delay": _env_float(
                "MAX_ENDPOINTING_DELAY", 0.9, min_value=0.1, max_value=4.0
            ),
            "alpha": _env_float("ENDPOINTING_ALPHA", 0.55, min_value=0.0, max_value=1.0),
        },
        interruption={
            "enabled": True,
            "mode": os.getenv("INTERRUPTION_MODE", "vad"),
            "min_duration": _env_float(
                "MIN_INTERRUPTION_DURATION", 0.22, min_value=0.05, max_value=2.0
            ),
            "min_words": _env_int("MIN_INTERRUPTION_WORDS", 0, min_value=0),
            "resume_false_interruption": True,
            "false_interruption_timeout": _env_float(
                "FALSE_INTERRUPTION_TIMEOUT", 1.0, min_value=0.0, max_value=5.0
            ),
        },
        preemptive_generation={
            "enabled": _env_bool("PREEMPTIVE_GENERATION", True),
            "preemptive_tts": _env_bool("PREEMPTIVE_TTS", True),
            "max_speech_duration": _env_float(
                "PREEMPTIVE_MAX_SPEECH_DURATION", 2.5, min_value=0.2, max_value=10.0
            ),
            "max_retries": _env_int("PREEMPTIVE_MAX_RETRIES", 1, min_value=0, max_value=5),
        },
    )
