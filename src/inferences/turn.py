from livekit.agents import TurnHandlingOptions, inference


def build_turn_handling() -> TurnHandlingOptions:
    return TurnHandlingOptions(
        turn_detection=inference.TurnDetector(),
        endpointing={
            "mode": "dynamic",
            "min_delay": 0.35,
            "max_delay": 1.2,
            "alpha": 0.65,
        },
        interruption={
            "enabled": True,
            "mode": "adaptive",
            "min_duration": 0.45,
            "min_words": 3,
            "discard_audio_if_uninterruptible": True,
            "false_interruption_timeout": 2.0,
            "resume_false_interruption": True,
        },
        preemptive_generation={
            "enabled": True,
            "preemptive_tts": False,
            "max_speech_duration": 2.5,
            "max_retries": 1,
        },
    )
