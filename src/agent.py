import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    ErrorEvent,
    JobContext,
    JobProcess,
    SessionUsageUpdatedEvent,
    TurnHandlingOptions,
    llm,
    room_io,
)
from livekit.plugins import ai_coustics, deepgram, elevenlabs, google, sarvam, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from tools import search_ai_mode, search_latest_news

logger = logging.getLogger("agent")
_DEFAULT_TURN_DETECTION = object()
INITIAL_GREETING_INSTRUCTIONS = """
Greet the user warmly in one natural breath, introduce yourself as Veena,
and ask what name you should call them. Sound human and present, not scripted.
Keep it to one or two short spoken sentences.
"""

env_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.local")
load_dotenv(env_file_path)

# LiveKit connection
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "my-agent")

# Provider API keys
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
sarvam_api_key = os.getenv("SARVAM_API_KEY")

# Model config
deepgram_model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
sarvam_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
sarvam_target_language_code = os.getenv("SARVAM_TARGET_LANGUAGE_CODE", "hi-IN")
sarvam_speaker = os.getenv("SARVAM_SPEAKER", "shubh")
gemini_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
gemini_thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low")
gemini_thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "-1")
gemini_fallback_model = os.getenv("GEMINI_FALLBACK_LLM_MODEL", "gemini-2.5-flash")


def _money(value: float) -> str:
    return f"${value:.6f}"


def _plugin_model(value: str, provider_prefix: str) -> str:
    model = value.strip().strip('"').strip("'")
    if model.startswith(f"{provider_prefix}/"):
        model = model.split("/", 1)[1]
    if provider_prefix == "deepgram" and model == "nova-3-general":
        return "nova-3"
    return model


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _use_elevenlabs_tts() -> bool:
    return _env_bool("USE_EL", True)


def _tts_provider(use_elevenlabs: bool | None = None) -> str:
    if use_elevenlabs is None:
        use_elevenlabs = _use_elevenlabs_tts()
    return "elevenlabs" if use_elevenlabs else "sarvam"


def _deepgram_language(use_elevenlabs: bool | None = None) -> str:
    if _tts_provider(use_elevenlabs) == "elevenlabs":
        return os.getenv("ELEVENLABS_DEEPGRAM_STT_LANGUAGE", "en")
    return os.getenv("DEEPGRAM_STT_LANGUAGE", "multi")


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = os.getenv(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value.strip())
        except ValueError:
            logger.warning("Invalid float for %s=%r; using %.3f", name, value, default)
            parsed = default

    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = os.getenv(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value.strip())
        except ValueError:
            logger.warning("Invalid int for %s=%r; using %d", name, value, default)
            parsed = default

    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _pricing_config() -> dict[str, float]:
    return {
        "deepgram_stt_per_minute": _env_float(
            "COST_DEEPGRAM_STT_PER_MINUTE_USD",
            0.0077,
            min_value=0.0,
        ),
        "gemini_input_per_1m": _env_float(
            "COST_GEMINI_INPUT_PER_1M_TOKENS_USD",
            0.10,
            min_value=0.0,
        ),
        "gemini_output_per_1m": _env_float(
            "COST_GEMINI_OUTPUT_PER_1M_TOKENS_USD",
            0.40,
            min_value=0.0,
        ),
        "sarvam_tts_per_1k_chars": _env_float(
            "COST_SARVAM_TTS_PER_1K_CHARS_USD",
            0.02,
            min_value=0.0,
        ),
        "elevenlabs_tts_per_1k_chars": _env_float(
            "COST_ELEVENLABS_TTS_PER_1K_CHARS_USD",
            0.05,
            min_value=0.0,
        ),
    }


def _usage_costs(
    usage: Any,
    pricing: dict[str, float],
    *,
    tts_provider: str = "sarvam",
) -> dict[str, float]:
    costs = {"deepgram": 0.0, "llm": 0.0, tts_provider: 0.0}

    if usage.type == "stt_usage":
        costs["deepgram"] = (
            usage.audio_duration / 60.0 * pricing["deepgram_stt_per_minute"]
        )
    elif usage.type == "llm_usage":
        billable_input_tokens = max(
            usage.input_tokens - usage.input_cached_tokens,
            0,
        )
        costs["llm"] = (
            billable_input_tokens / 1_000_000.0 * pricing["gemini_input_per_1m"]
            + usage.output_tokens / 1_000_000.0 * pricing["gemini_output_per_1m"]
        )
    elif usage.type == "tts_usage":
        costs[tts_provider] = (
            usage.characters_count
            / 1_000.0
            * pricing[f"{tts_provider}_tts_per_1k_chars"]
        )

    return costs


def _session_costs(
    session_usage: Any,
    pricing: dict[str, float],
    *,
    tts_provider: str = "sarvam",
) -> dict[str, float]:
    totals = {"deepgram": 0.0, "llm": 0.0, tts_provider: 0.0}

    for usage in session_usage.model_usage:
        for provider, cost in _usage_costs(
            usage,
            pricing,
            tts_provider=tts_provider,
        ).items():
            totals[provider] += cost

    totals["total"] = sum(totals.values())
    return totals


def _cost_delta(
    current: dict[str, float],
    previous: dict[str, float],
) -> dict[str, float]:
    keys = [
        key
        for key in ("deepgram", "llm", "sarvam", "elevenlabs", "total")
        if key in current or key in previous
    ]
    return {
        key: max(current.get(key, 0.0) - previous.get(key, 0.0), 0.0) for key in keys
    }


def _format_cost_summary(costs: dict[str, float]) -> str:
    parts = [f"deepgram={_money(costs['deepgram'])}", f"llm={_money(costs['llm'])}"]
    for provider in ("sarvam", "elevenlabs"):
        if provider in costs:
            parts.append(f"{provider}={_money(costs[provider])}")
    parts.append(f"total={_money(costs['total'])}")
    return " ".join(parts)


def _loggable_costs(costs: dict[str, float]) -> dict[str, str]:
    return {
        key: _money(costs[key])
        for key in ("deepgram", "llm", "sarvam", "elevenlabs", "total")
        if key in costs
    }


def _build_llm_kwargs(model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": google_api_key,
        "temperature": _env_float(
            "GEMINI_TEMPERATURE", 0.58, min_value=0.0, max_value=2.0
        ),
        "max_output_tokens": _env_int("GEMINI_MAX_OUTPUT_TOKENS", 1000, min_value=128),
    }

    if model.startswith("gemini-2.5"):
        kwargs["thinking_config"] = {
            "thinking_budget": _env_int(
                "GEMINI_THINKING_BUDGET",
                0,
                min_value=-1,
            )
        }
    elif model.startswith("gemini-3"):
        kwargs["thinking_config"] = {"thinking_level": gemini_thinking_level}

    return kwargs


def _build_llm(model: str) -> llm.LLM:
    primary = google.LLM(**_build_llm_kwargs(model))

    if not _env_bool("GEMINI_FALLBACK_ENABLED", True):
        return primary

    fallback_model = _plugin_model(gemini_fallback_model, "google")
    if fallback_model == model:
        return primary

    fallback = google.LLM(**_build_llm_kwargs(fallback_model))
    return llm.FallbackAdapter(
        [primary, fallback],
        attempt_timeout=_env_float(
            "GEMINI_FALLBACK_ATTEMPT_TIMEOUT", 4.0, min_value=1.0, max_value=60.0
        ),
        max_retry_per_llm=_env_int(
            "GEMINI_FALLBACK_MAX_RETRY_PER_LLM", 0, min_value=0, max_value=3
        ),
        retry_interval=_env_float(
            "GEMINI_FALLBACK_RETRY_INTERVAL", 0.12, min_value=0.0, max_value=5.0
        ),
    )


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
                "MIN_ENDPOINTING_DELAY", 0.32, min_value=0.05, max_value=2.0
            ),
            "max_delay": _env_float(
                "MAX_ENDPOINTING_DELAY", 1.15, min_value=0.1, max_value=4.0
            ),
            "alpha": _env_float(
                "ENDPOINTING_ALPHA", 0.55, min_value=0.0, max_value=1.0
            ),
        },
        interruption={
            "enabled": True,
            "mode": os.getenv("INTERRUPTION_MODE", "vad"),
            "min_duration": _env_float(
                "MIN_INTERRUPTION_DURATION", 0.30, min_value=0.05, max_value=2.0
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
                "PREEMPTIVE_MAX_SPEECH_DURATION", 1.8, min_value=0.2, max_value=10.0
            ),
            "max_retries": _env_int(
                "PREEMPTIVE_MAX_RETRIES", 1, min_value=0, max_value=5
            ),
        },
    )


def _build_turn_detector() -> EnglishModel | MultilingualModel:
    if _use_elevenlabs_tts():
        return EnglishModel()
    return MultilingualModel()


def _build_elevenlabs_tts(model: str) -> elevenlabs.TTS:
    return elevenlabs.TTS(
        model=model,
        voice_id=elevenlabs_voice_id,
        api_key=elevenlabs_api_key,
        language=os.getenv("ELEVENLABS_TTS_LANGUAGE", "en"),
        streaming_latency=_env_int(
            "ELEVENLABS_STREAMING_LATENCY", 3, min_value=0, max_value=4
        ),
        auto_mode=_env_bool("ELEVENLABS_AUTO_MODE", True),
        chunk_length_schedule=[
            _env_int("ELEVENLABS_CHUNK_1", 50, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_2", 80, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_3", 120, min_value=50, max_value=500),
            _env_int("ELEVENLABS_CHUNK_4", 160, min_value=50, max_value=500),
        ],
        voice_settings=elevenlabs.VoiceSettings(
            stability=_env_float(
                "ELEVENLABS_STABILITY", 0.38, min_value=0.0, max_value=1.0
            ),
            similarity_boost=_env_float(
                "ELEVENLABS_SIMILARITY_BOOST", 0.82, min_value=0.0, max_value=1.0
            ),
            speed=_env_float("ELEVENLABS_SPEED", 0.95, min_value=0.25, max_value=4.0),
            use_speaker_boost=_env_bool("ELEVENLABS_SPEAKER_BOOST", True),
        ),
        sync_alignment=_env_bool("ELEVENLABS_SYNC_ALIGNMENT", True),
    )


def _build_sarvam_tts(model: str) -> sarvam.TTS:
    return sarvam.TTS(
        model=model,
        target_language_code=sarvam_target_language_code,
        speaker=sarvam_speaker,
        api_key=sarvam_api_key,
        pace=_env_float("SARVAM_PACE", 1.0, min_value=0.5, max_value=2.0),
        temperature=_env_float(
            "SARVAM_TEMPERATURE", 0.6, min_value=0.01, max_value=1.0
        ),
        output_audio_bitrate=os.getenv("SARVAM_OUTPUT_AUDIO_BITRATE", "128k"),
        min_buffer_size=_env_int(
            "SARVAM_MIN_BUFFER_SIZE", 50, min_value=30, max_value=200
        ),
        max_chunk_length=_env_int(
            "SARVAM_MAX_CHUNK_LENGTH", 150, min_value=50, max_value=500
        ),
        speech_sample_rate=_env_int(
            "SARVAM_SPEECH_SAMPLE_RATE", 22050, min_value=8000, max_value=24000
        ),
    )


def _build_tts(model: str | None = None) -> elevenlabs.TTS | sarvam.TTS:
    if _use_elevenlabs_tts():
        selected_model = _plugin_model(
            model or os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
            "elevenlabs",
        )
        return _build_elevenlabs_tts(selected_model)

    selected_model = _plugin_model(model or sarvam_model, "sarvam")
    return _build_sarvam_tts(selected_model)


def _language_instructions() -> str:
    if _use_elevenlabs_tts():
        return """# Language
- Speak in English by default.
- If the user explicitly asks for another language, only switch when it is clear
  and natural for the configured voice.
- Keep names, product names, source names, and technical terms in English when
  translating them would sound unnatural.
- Do not announce that you are switching languages; just answer naturally."""

    return """# Language
- Speak in Hindi by default, using natural conversational Hindi.
- If the user speaks another language or explicitly asks for another language,
  respond in that language.
- Keep English names, product names, source names, and technical terms in English
  when translating them would sound unnatural.
- Do not announce that you are switching languages; just answer naturally."""


def _builtin_audio_clip(name: str, default: BuiltinAudioClip) -> BuiltinAudioClip:
    normalized = name.strip().upper()
    if normalized in BuiltinAudioClip.__members__:
        return BuiltinAudioClip[normalized]

    logger.warning("Unknown builtin audio clip %r; using %s", name, default.name)
    return default


def _build_background_audio_player() -> BackgroundAudioPlayer | None:
    if not _env_bool("BACKGROUND_AUDIO_ENABLED", True):
        return None

    ambient_clip = _builtin_audio_clip(
        os.getenv("BACKGROUND_AMBIENT_CLIP", "OFFICE_AMBIENCE"),
        BuiltinAudioClip.OFFICE_AMBIENCE,
    )
    ambient_sound = None
    if _env_bool("BACKGROUND_AMBIENT_SOUND_ENABLED", False):
        ambient_sound = AudioConfig(
            ambient_clip,
            volume=_env_float(
                "BACKGROUND_AMBIENT_VOLUME", 0.18, min_value=0.0, max_value=1.0
            ),
        )

    thinking_sound = None
    if _env_bool("BACKGROUND_THINKING_SOUND_ENABLED", True):
        thinking_clip = _builtin_audio_clip(
            os.getenv("BACKGROUND_THINKING_CLIP", "HOLD_MUSIC"),
            BuiltinAudioClip.HOLD_MUSIC,
        )
        thinking_sound = AudioConfig(
            thinking_clip,
            volume=_env_float(
                "BACKGROUND_THINKING_VOLUME", 0.10, min_value=0.0, max_value=1.0
            ),
        )
        if thinking_clip == BuiltinAudioClip.KEYBOARD_TYPING:
            thinking_sound = [
                AudioConfig(
                    thinking_clip,
                    volume=_env_float(
                        "BACKGROUND_THINKING_VOLUME",
                        0.10,
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    probability=0.75,
                ),
                AudioConfig(
                    BuiltinAudioClip.KEYBOARD_TYPING2,
                    volume=_env_float(
                        "BACKGROUND_THINKING_VOLUME_ALT",
                        0.08,
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    probability=0.25,
                ),
            ]

    return BackgroundAudioPlayer(
        ambient_sound=ambient_sound,
        thinking_sound=thinking_sound,
        stream_timeout_ms=_env_int(
            "BACKGROUND_AUDIO_STREAM_TIMEOUT_MS", 200, min_value=50, max_value=5000
        ),
    )


def _room_options() -> room_io.RoomOptions:
    audio_input = True
    if not _env_bool("ENABLE_NOISE_CANCELLATION", False):
        return room_io.RoomOptions(audio_input=audio_input)

    audio_input = room_io.AudioInputOptions(
        noise_cancellation=ai_coustics.audio_enhancement(
            model=ai_coustics.EnhancerModel.QUAIL_VF_L
        ),
    )
    return room_io.RoomOptions(audio_input=audio_input)


class AnchorVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""
You are Veena, a warm, natural, voice-first assistant.
You can help with news, AI updates, explanations, research, and everyday questions.

{_language_instructions()}

# Voice style
- When the session starts, greet the user and ask what you should call them.
- Speak in plain, natural language with a warm, human touch.
- Speak naturally like a calm human assistant.
- Do not over-explain unless the user asks.
- Sound warm, not robotic.
- Use small acknowledgements like "Got it", "Makes sense", "Sure", but do not repeat them too often.
- Keep answers brief by default, usually 1-3 sentences, but avoid sounding clipped
  or robotic.
- Use small conversational cues when helpful, like "got it", "fair question", or
  "that makes sense", but do not overdo filler.
- For the first turn, greet the user, introduce yourself, and ask what name you
  should call them.
- After the user gives their name, use it occasionally and naturally, not in every
  reply.
- Lead with the newest confirmed information, then add one useful detail.
- Do not use markdown, bullets, emojis, citations, or visual formatting in spoken replies.

# Tool use
- Use search_latest_news whenever the user asks about latest news, recent events,
  market-moving updates, sports results, public figures, products, laws, releases,
  or anything that might have changed recently.
- Use search_ai_mode for non-news web lookups, comparisons, explanations,
  recommendations, India-specific factual questions, local recommendations,
  government/services/process questions, prices, places, and general research
  that benefits from a synthesized answer.
- If a user asks about India or an Indian city and it is not latest news, use
  search_ai_mode with the most specific India location you can infer.
- When you use a search tool, call it directly without first saying a search
  status line. The tool itself says one short acknowledgement exactly once.
- While a search tool is running, stay silent and let the background thinking
  audio fill the wait. Do not repeat filler like "let me search" or "one moment".
- Summarize search results carefully. Mention source names and dates when available.
- If live search is unavailable, say that directly and answer only from stable knowledge.

# Boundaries
- Do not pretend to know real-time facts without searching.
- Keep speculation separate from confirmed results.
""",
            tools=[search_latest_news, search_ai_mode],
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    use_elevenlabs = _use_elevenlabs_tts()
    tts_provider = _tts_provider(use_elevenlabs)
    stt_language = _deepgram_language(use_elevenlabs)
    stt_model = _plugin_model(deepgram_model, "deepgram")
    llm_model = _plugin_model(gemini_model, "google")
    tts_model = _plugin_model(
        os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
        if use_elevenlabs
        else sarvam_model,
        tts_provider,
    )
    tts_language = (
        os.getenv("ELEVENLABS_TTS_LANGUAGE", "en")
        if use_elevenlabs
        else sarvam_target_language_code
    )
    tts_voice = elevenlabs_voice_id if use_elevenlabs else sarvam_speaker

    logger.info(
        "Starting low-latency voice pipeline with stt=%s:%s llm=%s tts_provider=%s tts=%s:%s voice=%s",
        stt_model,
        stt_language,
        llm_model,
        tts_provider,
        tts_model,
        tts_language,
        tts_voice,
    )

    session = AgentSession(
        stt=deepgram.STT(
            model=stt_model,
            language=stt_language,
            api_key=deepgram_api_key,
            interim_results=True,
            no_delay=True,
            endpointing_ms=_env_int(
                "DEEPGRAM_ENDPOINTING_MS", 90, min_value=10, max_value=500
            ),
            smart_format=_env_bool("DEEPGRAM_SMART_FORMAT", True),
            filler_words=_env_bool("DEEPGRAM_FILLER_WORDS", True),
        ),
        llm=_build_llm(llm_model),
        tts=_build_tts(tts_model),
        turn_handling=_build_turn_handling_options(),
        vad=ctx.proc.userdata["vad"],
        use_tts_aligned_transcript=_env_bool("USE_TTS_ALIGNED_TRANSCRIPT", False),
        min_consecutive_speech_delay=_env_float(
            "MIN_CONSECUTIVE_SPEECH_DELAY", 0.05, min_value=0.0, max_value=2.0
        ),
        aec_warmup_duration=_env_float(
            "AEC_WARMUP_DURATION", 0.2, min_value=0.0, max_value=5.0
        ),
        user_away_timeout=None,
    )

    pricing = _pricing_config()
    last_logged_costs = {
        "deepgram": 0.0,
        "llm": 0.0,
        tts_provider: 0.0,
        "total": 0.0,
    }

    @session.on("session_usage_updated")
    def _on_session_usage_updated(ev: SessionUsageUpdatedEvent):
        nonlocal last_logged_costs

        current_costs = _session_costs(
            ev.usage,
            pricing,
            tts_provider=tts_provider,
        )
        delta_costs = _cost_delta(current_costs, last_logged_costs)

        if delta_costs["llm"] == 0.0 and delta_costs[tts_provider] == 0.0:
            return

        last_logged_costs = current_costs

        logger.info(
            "Turn cost delta: %s | call total: %s",
            _format_cost_summary(delta_costs),
            _format_cost_summary(current_costs),
            extra={
                "cost_delta": _loggable_costs(delta_costs),
                "cost_total": _loggable_costs(current_costs),
            },
        )

    @session.on("error")
    def _on_error(ev: ErrorEvent):
        error_text = str(ev.error)
        if "no response generated" not in error_text:
            return

        logger.warning("Recovering from empty LLM response: %s", error_text)
        try:
            session.say(
                "I hit a temporary model hiccup. Please say that again.",
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )
        except RuntimeError:
            logger.warning("Could not speak LLM recovery message; session is closing")

    async def log_usage():
        final_costs = _session_costs(
            session.usage,
            pricing,
            tts_provider=tts_provider,
        )
        logger.info(
            "Session ended. Final call cost: %s",
            _format_cost_summary(final_costs),
            extra={"cost_total": _loggable_costs(final_costs)},
        )

    ctx.add_shutdown_callback(log_usage)

    background_audio = _build_background_audio_player()

    async def close_background_audio():
        if background_audio is not None:
            await background_audio.aclose()

    ctx.add_shutdown_callback(close_background_audio)

    await session.start(
        room=ctx.room,
        agent=AnchorVoiceAgent(),
        room_options=_room_options(),
    )

    if background_audio is not None:
        await background_audio.start(room=ctx.room, agent_session=session)

    await session.generate_reply(
        instructions=INITIAL_GREETING_INSTRUCTIONS,
        allow_interruptions=True,
        add_to_chat_ctx=True,
    )
    

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            ws_url=LIVEKIT_URL,
            agent_name=AGENT_NAME,
        )
    )
