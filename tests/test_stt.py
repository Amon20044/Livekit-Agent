import pytest

from app.models import VoiceConfig
from inferences.stt import build_stt, deepgram_language


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("multi", "multi"),
        ("en-IN", "en-IN"),
        ("hi-IN", "hi"),
        ("bn-IN", "bn"),
        ("ta-IN", "ta"),
        ("te-IN", "te"),
        ("kn-IN", "kn"),
        ("mr-IN", "mr"),
        ("gu-IN", "multi"),
        ("ml-IN", "multi"),
        ("pa-IN", "multi"),
        ("od-IN", "multi"),
    ],
)
def test_deepgram_language_uses_supported_nova_3_codes(
    configured: str, expected: str
) -> None:
    assert deepgram_language(configured) == expected


def test_build_stt_does_not_send_sarvam_locale_to_deepgram() -> None:
    engine = build_stt(VoiceConfig(language="hi-IN"), api_key="test-deepgram-key")

    assert str(engine._opts.language) == "hi"
