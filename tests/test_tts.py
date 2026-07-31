from app.models import VoiceConfig
from inferences.tts import build_tts


def test_build_tts_keeps_sarvam_configuration_minimal() -> None:
    engine = build_tts(
        VoiceConfig(
            language="hi-IN",
            tts_model="bulbul:v3",
            speaker="shubh",
            pace=1.1,
        ),
        api_key="test-sarvam-key",
    )

    assert engine._opts.model == "bulbul:v3"
    assert str(engine._opts.target_language_code) == "hi-IN"
    assert engine._opts.speaker == "shubh"
    assert engine._opts.pace == 1.1


def test_build_tts_defaults_multi_to_hindi_synthesis() -> None:
    engine = build_tts(VoiceConfig(language="multi"), api_key="test-sarvam-key")

    assert str(engine._opts.target_language_code) == "hi-IN"
