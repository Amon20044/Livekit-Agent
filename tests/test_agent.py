from types import SimpleNamespace

import pytest
from livekit.agents import BuiltinAudioClip, llm
from livekit.plugins import aws, elevenlabs, groq, sarvam

from app import agent_session as agent_module
from app.agent_session import WoiceVoiceAgent, _build_vad, _noise_cancellation_enabled
from audio.background import _build_background_audio_player
from core.env import _env_bool, _env_float, _env_int, _plugin_model
from inferences import llm as llm_module
from inferences import stt as stt_module
from inferences import tts as tts_module
from inferences import voice as voice_module
from inferences.llm import (
    _build_bedrock_llm,
    _build_groq_llm,
    _build_llm,
    _build_llm_kwargs,
    _LatencyOptimizedBedrockLLM,
    _llm_provider,
)
from inferences.tts import _build_tts
from inferences.turn import _build_turn_handling_options
from inferences.voice import _build_turn_detector, _deepgram_language
from prompts.instructions import (
    INITIAL_GREETING_INSTRUCTIONS,
    build_initial_greeting,
    build_returning_greeting,
)
from telemetry.costs import (
    _cost_delta,
    _format_cost_summary,
    _loggable_costs,
    _session_costs,
)
from tools import (
    get_dialed_phone_number,
    note_lead_progress,
    send_confirmed_lead_email_and_save,
)


def test_woice_agent_is_waitlist_intake_focused(monkeypatch) -> None:
    monkeypatch.setenv("USE_EL", "false")

    agent = WoiceVoiceAgent()

    assert "Woice AI" in agent.instructions
    assert "https://woice.vercel.app" in agent.instructions
    assert "waitlist" in agent.instructions.lower()
    assert "voice workflows" in agent.instructions
    assert "Hindi by default" in agent.instructions
    assert (
        "Do not save anything to Redis while the call is active" in agent.instructions
    )
    assert (
        "Should I add you to the Woice AI waitlist and send this recap to your email?"
        in agent.instructions
    )
    assert agent.tools == [
        send_confirmed_lead_email_and_save,
        get_dialed_phone_number,
        note_lead_progress,
    ]


def test_agent_instructions_cover_typed_input_and_readback(monkeypatch) -> None:
    monkeypatch.setenv("USE_EL", "false")

    instructions = WoiceVoiceAgent().instructions

    assert "type their email" in instructions
    assert "keypad" in instructions
    assert "get_dialed_phone_number" in instructions
    assert "one digit at a time" in instructions


def test_returning_greeting_variants(monkeypatch) -> None:
    monkeypatch.delenv("COMPANY_NAME", raising=False)

    assert build_returning_greeting(None) == build_initial_greeting()

    completed = build_returning_greeting({"status": "completed", "name": "Amon"})
    assert "Amon" in completed
    assert "anything more" in completed.lower()

    partial = build_returning_greeting(
        {"status": "partial", "name": "Amon", "email": "amon@example.com"}
    )
    assert "reconnect" in partial.lower()
    assert "amon@example.com" in partial


def test_company_name_is_configurable_via_env(monkeypatch) -> None:
    monkeypatch.setenv("USE_EL", "false")
    monkeypatch.setenv("COMPANY_NAME", "Acme Studio")
    monkeypatch.setenv("COMPANY_WEBSITE", "https://acme.example")

    agent = WoiceVoiceAgent()

    assert "Acme Studio" in agent.instructions
    assert "Woice AI" not in agent.instructions
    assert "Acme Studio" in build_initial_greeting()


@pytest.mark.asyncio
async def test_woice_agent_greets_and_asks_for_name_on_enter(monkeypatch) -> None:
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

    await WoiceVoiceAgent().on_enter()

    assert fake_session.generated_replies == [
        {
            "instructions": INITIAL_GREETING_INSTRUCTIONS,
            "allow_interruptions": True,
        }
    ]
    assert "Woice AI" in INITIAL_GREETING_INSTRUCTIONS
    assert "waitlist" in INITIAL_GREETING_INSTRUCTIONS.lower()


def test_woice_agent_uses_english_language_defaults_with_elevenlabs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("USE_EL", "true")

    agent = WoiceVoiceAgent()

    assert "English by default" in agent.instructions
    assert "Hindi by default" not in agent.instructions
    assert agent.tools == [
        send_confirmed_lead_email_and_save,
        get_dialed_phone_number,
        note_lead_progress,
    ]


def test_plugin_model_accepts_prefixed_and_legacy_values() -> None:
    assert _plugin_model("deepgram/nova-3-general", "deepgram") == "nova-3"
    assert _plugin_model("google/gemini-2.5-flash-lite", "google") == (
        "gemini-2.5-flash-lite"
    )
    assert _plugin_model("elevenlabs/eleven_flash_v2_5", "elevenlabs") == (
        "eleven_flash_v2_5"
    )
    assert _plugin_model("sarvam/bulbul:v3", "sarvam") == "bulbul:v3"


def test_build_stt_enables_smart_format_by_default(monkeypatch) -> None:
    captured: dict = {}

    class FakeSTT:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(stt_module.deepgram, "STT", FakeSTT)
    monkeypatch.delenv("DEEPGRAM_SMART_FORMAT", raising=False)

    stt_module.build_stt("nova-3", "multi")

    # smart_format makes spoken emails/phone numbers transcribe in written form.
    assert captured["smart_format"] is True
    assert captured["interim_results"] is True
    assert captured["no_delay"] is True


def test_build_stt_smart_format_can_be_disabled(monkeypatch) -> None:
    captured: dict = {}

    class FakeSTT:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(stt_module.deepgram, "STT", FakeSTT)
    monkeypatch.setenv("DEEPGRAM_SMART_FORMAT", "false")

    stt_module.build_stt("nova-3", "multi")

    assert captured["smart_format"] is False


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
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.delenv("MIN_ENDPOINTING_DELAY", raising=False)
    monkeypatch.delenv("MAX_ENDPOINTING_DELAY", raising=False)
    monkeypatch.delenv("INTERRUPTION_MODE", raising=False)
    monkeypatch.delenv("MIN_INTERRUPTION_DURATION", raising=False)
    monkeypatch.delenv("MIN_INTERRUPTION_WORDS", raising=False)
    monkeypatch.delenv("PREEMPTIVE_GENERATION", raising=False)
    monkeypatch.delenv("PREEMPTIVE_TTS", raising=False)

    options = _build_turn_handling_options(turn_detection=None)

    assert options["endpointing"]["mode"] == "dynamic"
    assert options["endpointing"]["min_delay"] == 0.22
    assert options["endpointing"]["max_delay"] == 0.9
    # VAD by default (adaptive needs LiveKit Cloud); 2 words required to interrupt
    # so single-word backchannel ("okay", "right") doesn't cut the agent off.
    assert options["interruption"]["mode"] == "vad"
    assert options["interruption"]["min_duration"] == 0.5
    assert options["interruption"]["min_words"] == 2
    assert options["preemptive_generation"]["enabled"] is True
    assert options["preemptive_generation"]["preemptive_tts"] is True
    assert options["preemptive_generation"]["max_speech_duration"] == 2.5


def test_turn_handling_defaults_to_adaptive_on_livekit_cloud(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://woice-prod.livekit.cloud")
    monkeypatch.delenv("INTERRUPTION_MODE", raising=False)

    options = _build_turn_handling_options(turn_detection=None)

    assert options["interruption"]["mode"] == "adaptive"


def test_turn_handling_min_words_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("MIN_INTERRUPTION_WORDS", "0")

    options = _build_turn_handling_options(turn_detection=None)

    assert options["interruption"]["min_words"] == 0


def test_turn_handling_allows_explicit_adaptive_interruption(monkeypatch) -> None:
    monkeypatch.setenv("INTERRUPTION_MODE", "adaptive")
    monkeypatch.setenv("MIN_INTERRUPTION_DURATION", "0.35")
    monkeypatch.setenv("MIN_INTERRUPTION_WORDS", "1")

    options = _build_turn_handling_options(turn_detection=None)

    assert options["interruption"]["mode"] == "adaptive"
    assert options["interruption"]["min_duration"] == 0.35
    assert options["interruption"]["min_words"] == 1


def test_turn_handling_invalid_interruption_mode_falls_back_to_vad(monkeypatch) -> None:
    monkeypatch.setenv("INTERRUPTION_MODE", "bad")

    options = _build_turn_handling_options(turn_detection=None)

    assert options["interruption"]["mode"] == "vad"


def test_vad_load_uses_fast_production_defaults(monkeypatch) -> None:
    captured = {}

    class FakeVAD:
        @staticmethod
        def load(**kwargs):
            captured.update(kwargs)
            return "vad"

    monkeypatch.setattr(agent_module.silero, "VAD", FakeVAD)
    monkeypatch.delenv("VAD_MIN_SPEECH_DURATION", raising=False)
    monkeypatch.delenv("VAD_MIN_SILENCE_DURATION", raising=False)
    monkeypatch.delenv("VAD_PREFIX_PADDING_DURATION", raising=False)
    monkeypatch.delenv("VAD_ACTIVATION_THRESHOLD", raising=False)

    assert _build_vad() == "vad"
    assert captured["min_speech_duration"] == 0.04
    assert captured["min_silence_duration"] == 0.35
    assert captured["prefix_padding_duration"] == 0.45
    assert captured["activation_threshold"] == 0.52
    assert captured["sample_rate"] == 16000
    assert captured["force_cpu"] is True


def test_noise_cancellation_defaults_to_livekit_cloud_only(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_NOISE_CANCELLATION", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    assert _noise_cancellation_enabled() is False

    monkeypatch.setenv("LIVEKIT_URL", "wss://woice-prod.livekit.cloud")
    assert _noise_cancellation_enabled() is True


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
