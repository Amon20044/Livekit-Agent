from types import SimpleNamespace

import pytest

from agent import (
    AnchorVoiceAgent,
    BuiltinAudioClip,
    _build_background_audio_player,
    _build_llm_kwargs,
    _build_turn_handling_options,
    _cost_delta,
    _env_bool,
    _env_float,
    _env_int,
    _format_cost_summary,
    _plugin_model,
    _session_costs,
)
from tools import search_ai_mode, search_latest_news


def test_anchor_agent_is_search_focused_and_has_serpapi_tools() -> None:
    agent = AnchorVoiceAgent()

    assert "Anchor" in agent.instructions
    assert "latest news" in agent.instructions
    assert "search_ai_mode" in agent.instructions
    assert "Yes, sure, let me search that for you" not in agent.instructions
    assert "The tool itself says one short acknowledgement exactly once" in (
        agent.instructions
    )
    assert "let the background thinking" in agent.instructions
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


def test_session_costs_include_deepgram_gemini_and_elevenlabs() -> None:
    pricing = {
        "deepgram_stt_per_minute": 0.0077,
        "gemini_input_per_1m": 0.10,
        "gemini_output_per_1m": 0.40,
        "elevenlabs_tts_per_1k_chars": 0.05,
    }
    usage = SimpleNamespace(
        model_usage=[
            SimpleNamespace(type="stt_usage", audio_duration=60.0),
            SimpleNamespace(
                type="llm_usage",
                input_tokens=1_000_000,
                input_cached_tokens=250_000,
                output_tokens=500_000,
            ),
            SimpleNamespace(type="tts_usage", characters_count=1_000),
        ]
    )

    costs = _session_costs(usage, pricing)

    assert costs["deepgram"] == pytest.approx(0.0077)
    assert costs["llm"] == pytest.approx(0.275)
    assert costs["elevenlabs"] == pytest.approx(0.05)
    assert costs["total"] == pytest.approx(0.3327)


def test_cost_delta_and_summary_format() -> None:
    current = {"deepgram": 0.02, "llm": 0.03, "elevenlabs": 0.04, "total": 0.09}
    previous = {"deepgram": 0.01, "llm": 0.01, "elevenlabs": 0.03, "total": 0.05}

    delta = _cost_delta(current, previous)

    assert delta == pytest.approx(
        {"deepgram": 0.01, "llm": 0.02, "elevenlabs": 0.01, "total": 0.04}
    )
    assert _format_cost_summary(delta) == (
        "deepgram=$0.010000 llm=$0.020000 elevenlabs=$0.010000 total=$0.040000"
    )


def test_background_audio_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("BACKGROUND_AUDIO_ENABLED", "false")

    assert _build_background_audio_player() is None


@pytest.mark.asyncio
async def test_background_audio_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("BACKGROUND_AUDIO_ENABLED", raising=False)
    monkeypatch.delenv("BACKGROUND_AMBIENT_SOUND_ENABLED", raising=False)
    monkeypatch.delenv("BACKGROUND_AMBIENT_CLIP", raising=False)
    monkeypatch.delenv("BACKGROUND_THINKING_CLIP", raising=False)

    player = _build_background_audio_player()

    assert player is not None
    assert player._ambient_sound is None
    assert player._thinking_sound.source == BuiltinAudioClip.HOLD_MUSIC
    await player.aclose()
