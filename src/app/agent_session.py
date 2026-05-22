import logging
import os

from livekit.agents import (
    Agent,
    AgentSession,
    ErrorEvent,
    JobContext,
    JobProcess,
    SessionUsageUpdatedEvent,
    room_io,
)
from livekit.plugins import ai_coustics, silero

from audio.background import _build_background_audio_player
from core.env import _env_bool, _env_float, _env_int, _plugin_model
from inferences.llm import _build_llm, _llm_provider
from inferences.stt import build_stt
from inferences.tts import _build_tts
from inferences.turn import _build_turn_handling_options
from inferences.voice import _deepgram_language, _tts_provider, _use_elevenlabs_tts
from prompts.instructions import INITIAL_GREETING_INSTRUCTIONS, build_agent_instructions
from settings import (
    bedrock_model,
    deepgram_model,
    elevenlabs_voice_id,
    gemini_model,
    groq_model,
    sarvam_model,
    sarvam_speaker,
    sarvam_target_language_code,
)
from telemetry.costs import (
    _cost_delta,
    _format_cost_summary,
    _loggable_costs,
    _pricing_config,
    _session_costs,
)
from tools import search_ai_mode, search_latest_news

logger = logging.getLogger("agent")


class AnchorVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_agent_instructions(_use_elevenlabs_tts()),
            tools=[search_latest_news],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=INITIAL_GREETING_INSTRUCTIONS,
            allow_interruptions=True,
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


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


async def entrypoint(ctx: JobContext):
    use_elevenlabs = _use_elevenlabs_tts()
    tts_provider = _tts_provider(use_elevenlabs)
    stt_language = _deepgram_language(use_elevenlabs)
    stt_model = _plugin_model(deepgram_model, "deepgram")
    llm_provider = _llm_provider()
    if llm_provider == "bedrock":
        llm_model = bedrock_model.strip()
    elif llm_provider == "groq":
        llm_model = _plugin_model(groq_model, "groq")
    else:
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
        "Starting low-latency voice pipeline with stt=%s:%s llm=%s:%s tts_provider=%s tts=%s:%s voice=%s",
        stt_model,
        stt_language,
        llm_provider,
        llm_model,
        tts_provider,
        tts_model,
        tts_language,
        tts_voice,
    )

    session = AgentSession(
        stt=build_stt(stt_model, stt_language),
        llm=_build_llm(llm_model),
        tts=_build_tts(tts_model),
        turn_handling=_build_turn_handling_options(),
        vad=ctx.proc.userdata["vad"],
        use_tts_aligned_transcript=_env_bool("USE_TTS_ALIGNED_TRANSCRIPT", False),
        min_consecutive_speech_delay=_env_float(
            "MIN_CONSECUTIVE_SPEECH_DELAY", 0.05, min_value=0.0, max_value=2.0
        ),
        aec_warmup_duration=_env_float(
            "AEC_WARMUP_DURATION", 0.1, min_value=0.0, max_value=5.0
        ),
        user_away_timeout=None,
    )

    @session.on("error")
    def _on_error(ev: ErrorEvent):
        error_text = str(ev.error)
        if "no response generated" not in error_text:
            return

        logger.warning("Recovering from empty LLM response: %s", error_text)
        try:
            session.say(
                "Sorry, say that again?",
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )
        except RuntimeError:
            logger.warning("Could not speak LLM recovery message; session is closing")

    # Cost accounting runs floating-point math and emits a log line on every usage
    # event, so it stays off the hot path by default. Enable COST_LOGGING_ENABLED
    # only when you need billing visibility.
    if _env_bool("COST_LOGGING_ENABLED", False):
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
