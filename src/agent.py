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
    JobContext,
    JobProcess,
    SessionUsageUpdatedEvent,
    TurnHandlingOptions,
    room_io,
)
from livekit.plugins import ai_coustics, deepgram, elevenlabs, google, silero
from livekit.plugins.turn_detector.english import EnglishModel

from tools import search_ai_mode, search_latest_news

logger = logging.getLogger("agent")
_DEFAULT_TURN_DETECTION = object()

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

# Model config
elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
deepgram_model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
deepgram_language = os.getenv("DEEPGRAM_STT_LANGUAGE", "en")
elevenlabs_model = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
elevenlabs_language = os.getenv("ELEVENLABS_TTS_LANGUAGE", "en")
gemini_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
gemini_thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low")
gemini_thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "0")


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
        "elevenlabs_tts_per_1k_chars": _env_float(
            "COST_ELEVENLABS_TTS_PER_1K_CHARS_USD",
            0.05,
            min_value=0.0,
        ),
    }


def _usage_costs(usage: Any, pricing: dict[str, float]) -> dict[str, float]:
    costs = {"deepgram": 0.0, "llm": 0.0, "elevenlabs": 0.0}

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
        costs["elevenlabs"] = (
            usage.characters_count / 1_000.0 * pricing["elevenlabs_tts_per_1k_chars"]
        )

    return costs


def _session_costs(session_usage: Any, pricing: dict[str, float]) -> dict[str, float]:
    totals = {"deepgram": 0.0, "llm": 0.0, "elevenlabs": 0.0}

    for usage in session_usage.model_usage:
        for provider, cost in _usage_costs(usage, pricing).items():
            totals[provider] += cost

    totals["total"] = sum(totals.values())
    return totals


def _cost_delta(
    current: dict[str, float],
    previous: dict[str, float],
) -> dict[str, float]:
    return {
        key: max(current.get(key, 0.0) - previous.get(key, 0.0), 0.0)
        for key in ("deepgram", "llm", "elevenlabs", "total")
    }


def _format_cost_summary(costs: dict[str, float]) -> str:
    return (
        f"deepgram={_money(costs['deepgram'])} "
        f"llm={_money(costs['llm'])} "
        f"elevenlabs={_money(costs['elevenlabs'])} "
        f"total={_money(costs['total'])}"
    )


def _loggable_costs(costs: dict[str, float]) -> dict[str, str]:
    return {
        key: _money(costs[key]) for key in ("deepgram", "llm", "elevenlabs", "total")
    }


def _build_llm_kwargs(model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": google_api_key,
        "temperature": _env_float(
            "GEMINI_TEMPERATURE", 0.35, min_value=0.0, max_value=2.0
        ),
        "max_output_tokens": _env_int("GEMINI_MAX_OUTPUT_TOKENS", 220, min_value=32),
    }

    if model.startswith("gemini-2.5"):
        kwargs["thinking_config"] = {
            "thinking_budget": _env_int(
                "GEMINI_THINKING_BUDGET",
                int(gemini_thinking_budget or "0"),
                min_value=0,
            )
        }
    elif model.startswith("gemini-3"):
        kwargs["thinking_config"] = {"thinking_level": gemini_thinking_level}

    return kwargs


def _build_turn_handling_options(
    turn_detection: Any = _DEFAULT_TURN_DETECTION,
) -> TurnHandlingOptions:
    if turn_detection is _DEFAULT_TURN_DETECTION:
        turn_detection = EnglishModel()

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
            "alpha": _env_float(
                "ENDPOINTING_ALPHA", 0.55, min_value=0.0, max_value=1.0
            ),
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
            "max_retries": _env_int(
                "PREEMPTIVE_MAX_RETRIES", 1, min_value=0, max_value=5
            ),
        },
    )


def _build_tts(model: str) -> elevenlabs.TTS:
    return elevenlabs.TTS(
        model=model,
        voice_id=elevenlabs_voice_id,
        api_key=elevenlabs_api_key,
        language=elevenlabs_language,
        streaming_latency=_env_int(
            "ELEVENLABS_STREAMING_LATENCY", 3, min_value=0, max_value=4
        ),
        auto_mode=_env_bool("ELEVENLABS_AUTO_MODE", True),
        chunk_length_schedule=[
            _env_int("ELEVENLABS_CHUNK_1", 50, min_value=50),
            _env_int("ELEVENLABS_CHUNK_2", 80, min_value=50),
            _env_int("ELEVENLABS_CHUNK_3", 120, min_value=50),
            _env_int("ELEVENLABS_CHUNK_4", 160, min_value=50),
        ],
        voice_settings=elevenlabs.VoiceSettings(
            stability=_env_float(
                "ELEVENLABS_STABILITY", 0.45, min_value=0.0, max_value=1.0
            ),
            similarity_boost=_env_float(
                "ELEVENLABS_SIMILARITY_BOOST", 0.75, min_value=0.0, max_value=1.0
            ),
            speed=_env_float("ELEVENLABS_SPEED", 1.08, min_value=0.7, max_value=1.2),
            use_speaker_boost=_env_bool("ELEVENLABS_SPEAKER_BOOST", False),
        ),
        sync_alignment=_env_bool("ELEVENLABS_SYNC_ALIGNMENT", False),
    )


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
            instructions="""
You are Anchor, a fast voice-first news and current-events agent.

# Voice style
- Speak in plain, natural language.
- Keep answers brief by default, usually 1-3 sentences.
- Lead with the newest confirmed information, then add one useful detail.
- Do not use markdown, bullets, emojis, citations, or visual formatting in spoken replies.

# Tool use
- Use search_latest_news whenever the user asks about latest news, recent events,
  market-moving updates, sports results, public figures, products, laws, releases,
  or anything that might have changed recently.
- Use search_ai_mode for non-news web lookups, comparisons, explanations,
  recommendations, and general research that benefits from a synthesized answer.
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
    stt_model = _plugin_model(deepgram_model, "deepgram")
    llm_model = _plugin_model(gemini_model, "google")
    tts_model = _plugin_model(elevenlabs_model, "elevenlabs")

    logger.info(
        "Starting low-latency voice pipeline with stt=%s llm=%s tts=%s voice=%s",
        stt_model,
        llm_model,
        tts_model,
        elevenlabs_voice_id,
    )

    session = AgentSession(
        stt=deepgram.STT(
            model=stt_model,
            language=deepgram_language,
            api_key=deepgram_api_key,
            interim_results=True,
            no_delay=True,
            endpointing_ms=_env_int(
                "DEEPGRAM_ENDPOINTING_MS", 25, min_value=10, max_value=500
            ),
            smart_format=_env_bool("DEEPGRAM_SMART_FORMAT", False),
            filler_words=_env_bool("DEEPGRAM_FILLER_WORDS", False),
        ),
        llm=google.LLM(**_build_llm_kwargs(llm_model)),
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
        "elevenlabs": 0.0,
        "total": 0.0,
    }

    @session.on("session_usage_updated")
    def _on_session_usage_updated(ev: SessionUsageUpdatedEvent):
        nonlocal last_logged_costs

        current_costs = _session_costs(ev.usage, pricing)
        delta_costs = _cost_delta(current_costs, last_logged_costs)

        if delta_costs["llm"] == 0.0 and delta_costs["elevenlabs"] == 0.0:
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

    async def log_usage():
        final_costs = _session_costs(session.usage, pricing)
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
