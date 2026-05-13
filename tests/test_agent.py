import pytest

from agent import (
    AnchorVoiceAgent,
    _build_background_audio_player,
    _build_llm_kwargs,
    _build_turn_handling_options,
    _env_bool,
    _env_float,
    _env_int,
    _plugin_model,
)
from tools import search_ai_mode, search_latest_news


def test_anchor_agent_is_search_focused_and_has_serpapi_tools() -> None:
    agent = AnchorVoiceAgent()

    assert "Anchor" in agent.instructions
    assert "latest news" in agent.instructions
    assert "search_ai_mode" in agent.instructions
    assert "Yes, sure, let me search that for you" not in agent.instructions
    assert "Never repeat or paraphrase it" in agent.instructions
    assert agent.tools == [search_latest_news, search_ai_mode]


def test_plugin_model_accepts_prefixed_and_legacy_values() -> None:
    assert _plugin_model("deepgram/nova-3-general", "deepgram") == "nova-3"
    assert _plugin_model("google/gemini-2.5-flash-lite", "google") == (
        "gemini-2.5-flash-lite"
    )
    assert _plugin_model("elevenlabs/eleven_flash_v2_5", "elevenlabs") == (
        "eleven_flash_v2_5"
    )


def test_env_bool(monkeypatch) -> None:
    monkeypatch.delenv("PREEMPTIVE_GENERATION", raising=False)
    assert _env_bool("PREEMPTIVE_GENERATION", True) is True

    monkeypatch.setenv("PREEMPTIVE_GENERATION", "false")
    assert _env_bool("PREEMPTIVE_GENERATION", True) is False

    monkeypatch.setenv("PREEMPTIVE_GENERATION", "yes")
    assert _env_bool("PREEMPTIVE_GENERATION", False) is True


def test_numeric_env_helpers_clamp_and_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MIN_ENDPOINTING_DELAY", "bad")
    assert _env_float("MIN_ENDPOINTING_DELAY", 0.22) == 0.22

    monkeypatch.setenv("ELEVENLABS_STREAMING_LATENCY", "9")
    assert _env_int("ELEVENLABS_STREAMING_LATENCY", 3, max_value=4) == 4


def test_turn_handling_defaults_are_low_latency(monkeypatch) -> None:
    monkeypatch.delenv("MIN_ENDPOINTING_DELAY", raising=False)
    monkeypatch.delenv("MAX_ENDPOINTING_DELAY", raising=False)
    monkeypatch.delenv("PREEMPTIVE_GENERATION", raising=False)
    monkeypatch.delenv("PREEMPTIVE_TTS", raising=False)

    options = _build_turn_handling_options(turn_detection=None)

    assert options["endpointing"]["mode"] == "dynamic"
    assert options["turn_detection"] is None
    assert options["endpointing"]["min_delay"] == 0.22
    assert options["endpointing"]["max_delay"] == 0.9
    assert options["preemptive_generation"]["enabled"] is True
    assert options["preemptive_generation"]["preemptive_tts"] is True


def test_gemini_25_defaults_to_no_extra_thinking(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)

    kwargs = _build_llm_kwargs("gemini-2.5-flash-lite")

    assert kwargs["thinking_config"] == {"thinking_budget": 0}
    assert kwargs["max_output_tokens"] == 220


def test_background_audio_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("BACKGROUND_AUDIO_ENABLED", "false")

    assert _build_background_audio_player() is None


@pytest.mark.asyncio
async def test_background_audio_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("BACKGROUND_AUDIO_ENABLED", raising=False)
    monkeypatch.delenv("BACKGROUND_AMBIENT_CLIP", raising=False)

    player = _build_background_audio_player()

    assert player is not None
    await player.aclose()
