from types import SimpleNamespace

import pytest

from app import agent_session as agent_module
from app.agent_session import AnchorVoiceAgent
from audio.background import _build_background_audio_player
from core.env import _env_bool, _env_float, _env_int, _plugin_model
from inferences import llm as llm_module
from inferences import tts as tts_module
from inferences import voice as voice_module
from inferences.llm import (
    _LatencyOptimizedBedrockLLM,
    _build_bedrock_llm,
    _build_groq_llm,
    _build_llm,
    _build_llm_kwargs,
    _llm_provider,
)
from inferences.turn import _build_turn_handling_options
from inferences.tts import _build_tts
from inferences.voice import _build_turn_detector, _deepgram_language
from livekit.agents import BuiltinAudioClip, llm
from livekit.plugins import aws, elevenlabs, groq, sarvam
from prompts.instructions import INITIAL_GREETING_INSTRUCTIONS
from telemetry.costs import (
    _cost_delta,
    _format_cost_summary,
    _loggable_costs,
    _session_costs,
)
from tools import search_ai_mode, search_latest_news


def test_anchor_agent_is_search_focused_and_has_serpapi_tools(monkeypatch) -> None:
    monkeypatch.setenv("USE_EL", "false")

    agent = AnchorVoiceAgent()

    assert "Veena" in agent.instructions
    assert "Hindi by default" in agent.instructions
    assert "search_latest_news" in agent.instructions
    assert "search_ai_mode" in agent.instructions
    # The agent must stay silent during tool calls (no spoken "let me search").
    assert 'Do NOT say "let me search"' in agent.instructions
    assert agent.tools == [search_latest_news, search_ai_mode]


@pytest.mark.asyncio
async def test_anchor_agent_greets_and_asks_for_name_on_enter(monkeypatch) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.generated_replies = []

        async def generate_reply(self, **kwargs) -> None:
            self.generated_replies.append(kwargs)

    fake_session = FakeSession()
    monkeypatch.setattr(
        agent_module.Agent,
        "session",
        property(lambda _agent: fake_session),
    )

    await AnchorVoiceAgent().on_enter()

    assert fake_session.generated_replies == [
        {
            "instructions": INITIAL_GREETING_INSTRUCTIONS,
            "allow_interruptions": True,
        }
    ]
    assert "ask what to call them" in INITIAL_GREETING_INSTRUCTIONS
    assert "Veena" in INITIAL_GREETING_INSTRUCTIONS


def test_anchor_agent_uses_english_language_defaults_with_elevenlabs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("USE_EL", "true")

    agent = AnchorVoiceAgent()

    assert "English by default" in agent.instructions
    assert "Hindi by default" not in agent.instructions
    assert agent.tools == [search_latest_news, search_ai_mode]


def test_plugin_model_accepts_prefixed_and_legacy_values() -> None:
    assert _plugin_model("deepgram/nova-3-general", "deepgram") == "nova-3"
    assert _plugin_model("google/gemini-2.5-flash-lite", "google") == (
        "gemini-2.5-flash-lite"
    )
    assert _plugin_model("elevenlabs/eleven_flash_v2_5", "elevenlabs") == (
        "eleven_flash_v2_5"
    )
    assert _plugin_model("sarvam/bulbul:v3", "sarvam") == "bulbul:v3"


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

    monkeypatch.setenv("SARVAM_MAX_CHUNK_LENGTH", "999")
    assert _env_int("SARVAM_MAX_CHUNK_LENGTH", 150, max_value=500) == 500


def test_turn_handling_defaults_are_low_latency(monkeypatch) -> None:
    monkeypatch.delenv("MIN_ENDPOINTING_DELAY", raising=False)
    monkeypatch.delenv("MAX_ENDPOINTING_DELAY", raising=False)
    monkeypatch.delenv("PREEMPTIVE_GENERATION", raising=False)
    monkeypatch.delenv("PREEMPTIVE_TTS", raising=False)

    options = _build_turn_handling_options(turn_detection=None)

    assert options["endpointing"]["mode"] == "dynamic"
    assert options["endpointing"]["min_delay"] == 0.22
    assert options["endpointing"]["max_delay"] == 0.9
    assert options["preemptive_generation"]["enabled"] is True
    assert options["preemptive_generation"]["preemptive_tts"] is True


def test_turn_handling_defaults_to_multilingual_detector(monkeypatch) -> None:
    class FakeMultilingualModel:
        pass

    monkeypatch.setattr(voice_module, "MultilingualModel", FakeMultilingualModel)
    monkeypatch.setenv("USE_EL", "false")

    options = _build_turn_handling_options()

    assert isinstance(options["turn_detection"], FakeMultilingualModel)


def test_use_el_defaults_speech_stack_to_english(monkeypatch) -> None:
    class FakeEnglishModel:
        pass

    monkeypatch.setenv("USE_EL", "true")
    monkeypatch.delenv("DEEPGRAM_STT_LANGUAGE", raising=False)
    monkeypatch.setattr(voice_module, "EnglishModel", FakeEnglishModel)

    assert _deepgram_language() == "en"
    assert isinstance(_build_turn_detector(), FakeEnglishModel)


def test_sarvam_defaults_speech_stack_to_multilingual(monkeypatch) -> None:
    class FakeMultilingualModel:
        pass

    monkeypatch.setenv("USE_EL", "false")
    monkeypatch.delenv("DEEPGRAM_STT_LANGUAGE", raising=False)
    monkeypatch.setattr(voice_module, "MultilingualModel", FakeMultilingualModel)

    assert _deepgram_language() == "multi"
    assert isinstance(_build_turn_detector(), FakeMultilingualModel)


def test_gemini_25_defaults_to_fast_thinking_and_compact_output(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)

    kwargs = _build_llm_kwargs("gemini-2.5-flash-lite")

    # thinking_budget=0 removes the pre-response reasoning delay for voice latency.
    assert kwargs["thinking_config"] == {"thinking_budget": 0}
    assert kwargs["max_output_tokens"] == 220


def test_llm_uses_fallback_adapter_by_default(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.delenv("GEMINI_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_LLM_MODEL", raising=False)

    model = _build_llm("gemini-2.5-flash-lite")

    assert isinstance(model, llm.FallbackAdapter)


def test_llm_fallback_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "false")

    model = _build_llm("gemini-2.5-flash-lite")

    assert not isinstance(model, llm.FallbackAdapter)


def test_llm_provider_switch_maps_aliases(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "google")
    assert _llm_provider() == "google"

    monkeypatch.setenv("LLM_PROVIDER", "aws")
    assert _llm_provider() == "bedrock"

    monkeypatch.setenv("LLM_PROVIDER", "BEDROCK")
    assert _llm_provider() == "bedrock"

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert _llm_provider() == "groq"

    monkeypatch.setenv("LLM_PROVIDER", "GROQ")
    assert _llm_provider() == "groq"

    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    assert _llm_provider() == "google"


def test_build_llm_switches_to_bedrock_when_selected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "aws")
    monkeypatch.setenv("BEDROCK_LATENCY_OPTIMIZED", "false")

    model = _build_llm("amazon.nova-micro-v1:0")

    assert isinstance(model, aws.LLM)
    assert model.model == "amazon.nova-micro-v1:0"


def test_bedrock_latency_optimized_toggle(monkeypatch) -> None:
    monkeypatch.setenv("BEDROCK_LATENCY_OPTIMIZED", "true")
    optimized = _build_bedrock_llm("us.anthropic.claude-3-5-haiku-20241022-v1:0")
    assert isinstance(optimized, _LatencyOptimizedBedrockLLM)

    monkeypatch.setenv("BEDROCK_LATENCY_OPTIMIZED", "false")
    plain = _build_bedrock_llm("amazon.nova-micro-v1:0")
    assert isinstance(plain, aws.LLM)
    assert not isinstance(plain, _LatencyOptimizedBedrockLLM)


def test_latency_optimized_injects_performance_config(monkeypatch) -> None:
    monkeypatch.setattr(
        aws.LLM, "chat", lambda self, **kwargs: SimpleNamespace(_opts={})
    )

    optimized = _LatencyOptimizedBedrockLLM(model="amazon.nova-micro-v1:0")
    stream = optimized.chat(chat_ctx=None)

    assert stream._opts["performanceConfig"] == {"latency": "optimized"}


def test_build_llm_switches_to_groq_when_selected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm_module, "groq_api_key", "test-key")

    model = _build_llm("llama-3.1-8b-instant")

    assert isinstance(model, groq.LLM)
    assert model.model == "llama-3.1-8b-instant"


def test_build_groq_llm_applies_fast_defaults(monkeypatch) -> None:
    monkeypatch.setattr(llm_module, "groq_api_key", "test-key")
    monkeypatch.delenv("GROQ_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GROQ_TEMPERATURE", raising=False)

    model = _build_groq_llm("llama-3.1-8b-instant")

    assert isinstance(model, groq.LLM)
    assert model.model == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_use_el_builds_optimized_elevenlabs_english_tts(monkeypatch) -> None:
    monkeypatch.setenv("USE_EL", "true")
    monkeypatch.delenv("ELEVENLABS_TTS_MODEL", raising=False)
    monkeypatch.delenv("ELEVENLABS_TTS_LANGUAGE", raising=False)
    monkeypatch.setattr(tts_module, "elevenlabs_api_key", "test-key")
    monkeypatch.setattr(tts_module, "elevenlabs_voice_id", "test-voice")

    tts = _build_tts()

    assert isinstance(tts, elevenlabs.TTS)
    assert tts._opts.model == "eleven_flash_v2_5"
    assert str(tts._opts.language) == "en"
    assert tts._opts.voice_id == "test-voice"
    assert tts._opts.streaming_latency == 3
    assert tts._opts.auto_mode is True
    assert tts._opts.chunk_length_schedule == [50, 70, 100, 140]
    assert tts._opts.sync_alignment is False
    assert tts._opts.voice_settings.stability == 0.45
    assert tts._opts.voice_settings.similarity_boost == 0.75
    assert tts._opts.voice_settings.speed == 1.08
    assert tts._opts.voice_settings.use_speaker_boost is False
    await tts.aclose()


def test_sarvam_tts_uses_multilingual_indian_defaults(monkeypatch) -> None:
    monkeypatch.setenv("USE_EL", "false")
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.delenv("SARVAM_TARGET_LANGUAGE_CODE", raising=False)
    monkeypatch.delenv("SARVAM_SPEAKER", raising=False)

    tts = _build_tts()

    assert isinstance(tts, sarvam.TTS)
    assert tts._opts.model == "bulbul:v3"
    assert tts._opts.target_language_code == "hi-IN"
    assert tts._opts.speaker == "shubh"


def test_session_costs_include_deepgram_gemini_and_sarvam() -> None:
    pricing = {
        "deepgram_stt_per_minute": 0.0077,
        "gemini_input_per_1m": 0.10,
        "gemini_output_per_1m": 0.40,
        "sarvam_tts_per_1k_chars": 0.02,
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

    costs = _session_costs(usage, pricing, tts_provider="sarvam")

    assert costs["deepgram"] == pytest.approx(0.0077)
    assert costs["llm"] == pytest.approx(0.275)
    assert costs["sarvam"] == pytest.approx(0.02)
    assert costs["total"] == pytest.approx(0.3027)


def test_session_costs_include_deepgram_gemini_and_elevenlabs() -> None:
    pricing = {
        "deepgram_stt_per_minute": 0.0077,
        "gemini_input_per_1m": 0.10,
        "gemini_output_per_1m": 0.40,
        "sarvam_tts_per_1k_chars": 0.02,
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

    costs = _session_costs(usage, pricing, tts_provider="elevenlabs")

    assert costs["deepgram"] == pytest.approx(0.0077)
    assert costs["llm"] == pytest.approx(0.275)
    assert costs["elevenlabs"] == pytest.approx(0.05)
    assert costs["total"] == pytest.approx(0.3327)


def test_cost_delta_and_summary_format() -> None:
    current = {"deepgram": 0.02, "llm": 0.03, "sarvam": 0.04, "total": 0.09}
    previous = {"deepgram": 0.01, "llm": 0.01, "sarvam": 0.03, "total": 0.05}

    delta = _cost_delta(current, previous)

    assert delta == pytest.approx(
        {"deepgram": 0.01, "llm": 0.02, "sarvam": 0.01, "total": 0.04}
    )
    assert _format_cost_summary(delta) == (
        "deepgram=$0.010000 llm=$0.020000 sarvam=$0.010000 total=$0.040000"
    )
    assert _loggable_costs(delta) == {
        "deepgram": "$0.010000",
        "llm": "$0.020000",
        "sarvam": "$0.010000",
        "total": "$0.040000",
    }


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
