from types import SimpleNamespace

import pytest
from livekit.agents import BuiltinAudioClip, llm
from livekit.plugins import ai_coustics, aws, elevenlabs, groq, sarvam

from app import agent_session as agent_module
from app.agent_session import (
    WoiceVoiceAgent,
    _aicoustics_auth,
    _aicoustics_license_key,
    _build_vad,
    _noise_cancellation_enabled,
)
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
    _use_vertexai,
)
from inferences.tts import _build_tts
from inferences.turn import _build_turn_handling_options
from inferences.voice import _build_turn_detector, _stt_language
from prompts.instructions import (
    INITIAL_GREETING_INSTRUCTIONS,
    build_agent_instructions,
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


def test_returning_greeting_is_warm_and_dynamic(monkeypatch) -> None:
    monkeypatch.delenv("COMPANY_NAME", raising=False)

    completed = build_returning_greeting(
        {"status": "completed", "name": "Amon"}
    ).lower()
    # Warm + personal + non-scripted: greet by name, ask how they are, vary wording.
    assert "amon" in completed
    assert "how have you been" in completed
    assert "vary your wording" in completed
    assert "scripted" in completed

    partial = build_returning_greeting(
        {"status": "partial", "name": "Amon", "email": "amon@example.com"}
    ).lower()
    assert "welcome back" in partial
    assert "vary your wording" in partial


def test_email_capture_forbids_invented_separators() -> None:
    instructions = build_agent_instructions(use_elevenlabs=True).lower()

    # The dots in "amon.sharma.2000" came from the LLM, not the normalizer, so the
    # prompt must explicitly forbid inserting separators the caller didn't speak.
    assert "amonsharma2000@gmail.com" in instructions
    assert "did not actually say" in instructions
    assert "firstname.lastname" in instructions


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

    # The opening line is uninterruptible: at call start the acoustic echo
    # canceller has not converged, so an interruptible greeting hears its own
    # echo and cuts itself off ("no sound at the start"). Locking the first turn
    # guarantees the full greeting plays; normal barge-in resumes afterward.
    assert fake_session.generated_replies == [
        {
            "instructions": INITIAL_GREETING_INSTRUCTIONS,
            "allow_interruptions": False,
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


def test_build_stt_uses_speechmatics_defaults_from_reference_config(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeSTT:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(stt_module.speechmatics, "STT", FakeSTT)
    monkeypatch.delenv("SPEECHMATICS_OPERATING_POINT", raising=False)
    monkeypatch.delenv("SPEECHMATICS_INCLUDE_PARTIALS", raising=False)
    monkeypatch.delenv("SPEECHMATICS_MAX_DELAY", raising=False)
    monkeypatch.delenv("SPEECHMATICS_END_OF_UTTERANCE_SILENCE_TRIGGER", raising=False)
    monkeypatch.setenv("SPEECHMATICS_TURN_DETECTION_MODE", "fixed")
    monkeypatch.delenv("SPEECHMATICS_ENABLE_DIARIZATION", raising=False)
    monkeypatch.delenv("SPEECHMATICS_PREFER_CURRENT_SPEAKER", raising=False)
    monkeypatch.delenv("SPEECHMATICS_SPEAKER_SENSITIVITY", raising=False)
    monkeypatch.delenv("SPEECHMATICS_MAX_SPEAKERS", raising=False)

    stt_module.build_stt("en")

    assert captured["language"] == "en"
    assert (
        captured["operating_point"] == stt_module.speechmatics.OperatingPoint.ENHANCED
    )
    assert captured["include_partials"] is True
    assert captured["max_delay"] == 0.7
    assert captured["end_of_utterance_silence_trigger"] == 0.5
    assert (
        captured["turn_detection_mode"]
        == stt_module.speechmatics.TurnDetectionMode.SMART_TURN
    )
    assert captured["enable_diarization"] is True
    assert captured["speaker_active_format"] == "<{speaker_id}>{text}</{speaker_id}>"
    assert captured["speaker_passive_format"] == "[{speaker_id}^PASSIVE*] {text}"
    # Locks STT onto the customer; optional diarization knobs stay unset by default.
    assert captured["prefer_current_speaker"] is True
    assert "speaker_sensitivity" not in captured
    assert "max_speakers" not in captured
    assert [
        (entry.content, entry.sounds_like) for entry in captured["additional_vocab"]
    ] == [
        ("LiveKit", ["live kit"]),
        ("Woice", ["voice"]),
        ("2000", ["two thousands"]),
        ("Speechmatics", ["speech-matics"]),
    ]


def test_build_stt_speaker_focus_knobs_are_configurable(monkeypatch) -> None:
    captured: dict = {}

    class FakeSTT:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(stt_module.speechmatics, "STT", FakeSTT)
    monkeypatch.setenv("SPEECHMATICS_PREFER_CURRENT_SPEAKER", "false")
    monkeypatch.setenv("SPEECHMATICS_SPEAKER_SENSITIVITY", "0.4")
    monkeypatch.setenv("SPEECHMATICS_MAX_SPEAKERS", "2")

    stt_module.build_stt("en")

    assert captured["prefer_current_speaker"] is False
    assert captured["speaker_sensitivity"] == 0.4
    assert captured["max_speakers"] == 2


def test_build_stt_speechmatics_latency_and_diarization_are_configurable(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeSTT:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(stt_module.speechmatics, "STT", FakeSTT)
    monkeypatch.setenv("SPEECHMATICS_OPERATING_POINT", "standard")
    monkeypatch.setenv("SPEECHMATICS_INCLUDE_PARTIALS", "false")
    monkeypatch.setenv("SPEECHMATICS_MAX_DELAY", "1.1")
    monkeypatch.setenv("SPEECHMATICS_END_OF_UTTERANCE_SILENCE_TRIGGER", "0.8")
    monkeypatch.setenv("SPEECHMATICS_ENABLE_DIARIZATION", "false")

    stt_module.build_stt("en")

    assert (
        captured["operating_point"] == stt_module.speechmatics.OperatingPoint.STANDARD
    )
    assert captured["include_partials"] is False
    assert captured["max_delay"] == 1.1
    assert captured["end_of_utterance_silence_trigger"] == 0.8
    assert (
        captured["turn_detection_mode"]
        == stt_module.speechmatics.TurnDetectionMode.SMART_TURN
    )
    assert captured["enable_diarization"] is False


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
    # VAD by default (adaptive needs LiveKit Cloud); 6 words required to interrupt
    # so backchannel and short fillers (up to 5 words) don't cut the agent off.
    assert options["interruption"]["mode"] == "vad"
    assert options["interruption"]["min_duration"] == 0.5
    assert options["interruption"]["min_words"] == 6
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
    monkeypatch.delenv("AICOUSTICS_LICENSE_KEY", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    assert _noise_cancellation_enabled() is False

    monkeypatch.setenv("LIVEKIT_URL", "wss://woice-prod.livekit.cloud")
    assert _noise_cancellation_enabled() is True


def test_noise_cancellation_auto_enables_with_license_key(monkeypatch) -> None:
    # A direct ai-coustics license key lets NC run on self-hosted/localhost, so it
    # should auto-enable even when the URL is not LiveKit Cloud.
    monkeypatch.delenv("ENABLE_NOISE_CANCELLATION", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("AICOUSTICS_LICENSE_KEY", "lic-123")
    assert _noise_cancellation_enabled() is True


def test_aicoustics_auth_uses_license_key_then_cloud(monkeypatch) -> None:
    monkeypatch.setenv("AICOUSTICS_LICENSE_KEY", "lic-123")
    assert _aicoustics_license_key() == "lic-123"
    license_auth = _aicoustics_auth()
    assert isinstance(license_auth, ai_coustics.AiCousticsApi)
    assert license_auth.provider == "aiCousticsApi"

    monkeypatch.delenv("AICOUSTICS_LICENSE_KEY", raising=False)
    assert _aicoustics_license_key() is None
    assert _aicoustics_auth().provider == "livekitCloud"


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
    monkeypatch.delenv("SPEECHMATICS_STT_LANGUAGE", raising=False)
    monkeypatch.delenv("ELEVENLABS_SPEECHMATICS_STT_LANGUAGE", raising=False)
    monkeypatch.setattr(voice_module, "EnglishModel", FakeEnglishModel)

    assert _stt_language() == "en"
    assert isinstance(_build_turn_detector(), FakeEnglishModel)


def test_sarvam_defaults_speechmatics_stack_to_english(monkeypatch) -> None:
    class FakeMultilingualModel:
        pass

    monkeypatch.setenv("USE_EL", "false")
    monkeypatch.delenv("SPEECHMATICS_STT_LANGUAGE", raising=False)
    monkeypatch.setattr(voice_module, "MultilingualModel", FakeMultilingualModel)

    assert _stt_language() == "en"
    assert isinstance(_build_turn_detector(), FakeMultilingualModel)


def test_speechmatics_language_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("USE_EL", "false")
    monkeypatch.setenv("SPEECHMATICS_STT_LANGUAGE", "hi")

    assert _stt_language() == "hi"


def test_gemini_25_defaults_to_fast_thinking_and_compact_output(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)

    kwargs = _build_llm_kwargs("gemini-2.5-flash-lite")

    # thinking_budget=0 removes the pre-response reasoning delay for voice latency.
    assert kwargs["thinking_config"] == {"thinking_budget": 0}
    assert kwargs["max_output_tokens"] == 220


def test_gemini_api_kwargs_send_api_key_not_vertexai(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setattr(llm_module, "google_api_key", "test-key")

    kwargs = _build_llm_kwargs("gemini-2.5-flash-lite")

    assert not _use_vertexai()
    assert kwargs["api_key"] == "test-key"
    assert "vertexai" not in kwargs
    assert "project" not in kwargs


def test_vertexai_kwargs_use_adc_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "woice-prod")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-south1")

    kwargs = _build_llm_kwargs("gemini-2.5-flash-lite")

    # Vertex AI authenticates via ADC; the plugin nulls any key, so we never send one.
    assert _use_vertexai()
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "woice-prod"
    assert kwargs["location"] == "asia-south1"
    assert "api_key" not in kwargs


def test_vertexai_infers_project_and_defaults_location(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    kwargs = _build_llm_kwargs("gemini-2.5-flash-lite")

    # Project omitted so the plugin infers it from ADC; location falls back.
    assert kwargs["vertexai"] is True
    assert "project" not in kwargs
    assert kwargs["location"] == "us-central1"
    assert "api_key" not in kwargs


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
    # PCM output (no MP3 decode) and a warm WebSocket are the packet-loss levers.
    assert tts._opts.encoding == "pcm_24000"
    assert tts._opts.sample_rate == 24000
    assert tts._opts.inactivity_timeout == 180
    assert tts._opts.apply_text_normalization == "auto"
    assert tts._opts.streaming_latency == 3
    assert tts._opts.auto_mode is True
    assert tts._opts.chunk_length_schedule == [50, 70, 100, 140]
    assert tts._opts.sync_alignment is False
    assert tts._opts.voice_settings.stability == 0.45
    assert tts._opts.voice_settings.similarity_boost == 0.75
    assert tts._opts.voice_settings.speed == 1.08
    assert tts._opts.voice_settings.use_speaker_boost is False
    await tts.aclose()


def test_elevenlabs_encoding_defaults_to_pcm_and_validates(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_OUTPUT_FORMAT", raising=False)
    assert tts_module._elevenlabs_encoding() == "pcm_24000"

    monkeypatch.setenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
    assert tts_module._elevenlabs_encoding() == "pcm_16000"

    # Unsupported formats fall back to the safe PCM default instead of crashing.
    monkeypatch.setenv("ELEVENLABS_OUTPUT_FORMAT", "flac_9000")
    assert tts_module._elevenlabs_encoding() == "pcm_24000"


def test_elevenlabs_text_normalization_validates(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_TEXT_NORMALIZATION", raising=False)
    assert tts_module._elevenlabs_text_normalization() == "auto"

    monkeypatch.setenv("ELEVENLABS_TEXT_NORMALIZATION", "off")
    assert tts_module._elevenlabs_text_normalization() == "off"

    monkeypatch.setenv("ELEVENLABS_TEXT_NORMALIZATION", "loud")
    assert tts_module._elevenlabs_text_normalization() == "auto"


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


def test_session_costs_include_speechmatics_gemini_and_sarvam() -> None:
    pricing = {
        "speechmatics_stt_per_minute": 0.01,
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

    assert costs["speechmatics"] == pytest.approx(0.01)
    assert costs["llm"] == pytest.approx(0.275)
    assert costs["sarvam"] == pytest.approx(0.02)
    assert costs["total"] == pytest.approx(0.305)


def test_session_costs_include_speechmatics_gemini_and_elevenlabs() -> None:
    pricing = {
        "speechmatics_stt_per_minute": 0.01,
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

    assert costs["speechmatics"] == pytest.approx(0.01)
    assert costs["llm"] == pytest.approx(0.275)
    assert costs["elevenlabs"] == pytest.approx(0.05)
    assert costs["total"] == pytest.approx(0.335)


def test_cost_delta_and_summary_format() -> None:
    current = {"speechmatics": 0.02, "llm": 0.03, "sarvam": 0.04, "total": 0.09}
    previous = {"speechmatics": 0.01, "llm": 0.01, "sarvam": 0.03, "total": 0.05}

    delta = _cost_delta(current, previous)

    assert delta == pytest.approx(
        {"speechmatics": 0.01, "llm": 0.02, "sarvam": 0.01, "total": 0.04}
    )
    assert _format_cost_summary(delta) == (
        "speechmatics=$0.010000 llm=$0.020000 sarvam=$0.010000 total=$0.040000"
    )
    assert _loggable_costs(delta) == {
        "speechmatics": "$0.010000",
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
