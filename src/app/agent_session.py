import asyncio
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

from app.dtmf import DtmfCollector, register_dtmf_collector
from audio.background import _build_background_audio_player
from core.env import _env_bool, _env_float, _env_int, _plugin_model
from inferences.llm import _build_llm, _llm_provider
from inferences.stt import _stt_provider_name, build_stt, stt_model_name
from inferences.tts import _build_tts
from inferences.turn import _build_turn_handling_options
from inferences.voice import _stt_language, _tts_provider, _use_elevenlabs_tts
from prompts.instructions import build_agent_instructions, build_returning_greeting
from settings import (
    bedrock_model,
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
from tools import (
    get_dialed_phone_number,
    note_lead_progress,
    send_confirmed_lead_email_and_save,
)
from tools.company import (
    caller_ref_from_room,
    lookup_caller,
    redis_health_check,
    upsert_caller_checkpoint,
)

logger = logging.getLogger("agent")


class WoiceVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_agent_instructions(_use_elevenlabs_tts()),
            tools=[
                send_confirmed_lead_email_and_save,
                get_dialed_phone_number,
                note_lead_progress,
            ],
        )

    async def on_enter(self) -> None:
        userdata = getattr(self.session, "userdata", None)
        record = (
            userdata.get("returning_caller") if isinstance(userdata, dict) else None
        )
        # Opening line is uninterruptible. At call start the echo canceller has
        # not converged yet, so an interruptible greeting hears its own audio
        # echo back and self-interrupts within the first frames — the caller
        # hears a chopped intro or "no sound." Locking the first turn guarantees
        # the greeting plays in full; barge-in resumes for the rest of the call.
        await self.session.generate_reply(
            instructions=build_returning_greeting(record),
            allow_interruptions=False,
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = _build_vad()


def _build_vad():
    return silero.VAD.load(
        min_speech_duration=_env_float(
            "VAD_MIN_SPEECH_DURATION", 0.04, min_value=0.01, max_value=0.5
        ),
        min_silence_duration=_env_float(
            "VAD_MIN_SILENCE_DURATION", 0.42, min_value=0.1, max_value=1.2
        ),
        prefix_padding_duration=_env_float(
            "VAD_PREFIX_PADDING_DURATION", 0.45, min_value=0.0, max_value=1.0
        ),
        max_buffered_speech=_env_float(
            "VAD_MAX_BUFFERED_SPEECH", 45.0, min_value=5.0, max_value=120.0
        ),
        activation_threshold=_env_float(
            "VAD_ACTIVATION_THRESHOLD", 0.52, min_value=0.1, max_value=0.95
        ),
        sample_rate=_vad_sample_rate(),
        force_cpu=_env_bool("VAD_FORCE_CPU", True),
    )


def _livekit_cloud_transport() -> bool:
    return ".livekit.cloud" in (os.getenv("LIVEKIT_URL") or "").lower()


def _aicoustics_license_key() -> str | None:
    """Direct ai-coustics license key, which lets NC run without LiveKit Cloud."""
    key = (os.getenv("AICOUSTICS_LICENSE_KEY") or "").strip()
    return key or None


def _noise_cancellation_enabled() -> bool:
    # ai-coustics input enhancement removes background noise *before* STT, so the
    # transcriber stops hearing the room. It needs either LiveKit Cloud (cloud
    # billing) or a direct ai-coustics license key, so auto-on only when one of
    # those is available; otherwise it would fail to authenticate. Override with
    # ENABLE_NOISE_CANCELLATION.
    default_on = _livekit_cloud_transport() or _aicoustics_license_key() is not None
    return _env_bool("ENABLE_NOISE_CANCELLATION", default_on)


def _aicoustics_auth() -> ai_coustics.AuthBase:
    """Use the direct license key when set, else fall back to LiveKit Cloud auth."""
    key = _aicoustics_license_key()
    if key:
        return ai_coustics.Auth.ai_coustics_api(license_key=key)
    return ai_coustics.Auth.livekit_cloud()


def _noise_cancellation_model() -> ai_coustics.EnhancerModel:
    model_name = os.getenv("NOISE_CANCELLATION_MODEL", "QUAIL_VF_L").strip().upper()
    return getattr(
        ai_coustics.EnhancerModel,
        model_name,
        ai_coustics.EnhancerModel.QUAIL_VF_L,
    )


def _vad_sample_rate() -> int:
    configured = _env_int("VAD_SAMPLE_RATE", 16000, min_value=8000, max_value=16000)
    return 8000 if configured == 8000 else 16000


def _room_options() -> room_io.RoomOptions:
    audio_input = True
    if not _noise_cancellation_enabled():
        return room_io.RoomOptions(audio_input=audio_input)

    audio_input = room_io.AudioInputOptions(
        noise_cancellation=ai_coustics.audio_enhancement(
            model=_noise_cancellation_model(),
            auth=_aicoustics_auth(),
        ),
    )
    return room_io.RoomOptions(audio_input=audio_input)


async def entrypoint(ctx: JobContext):
    use_elevenlabs = _use_elevenlabs_tts()
    tts_provider = _tts_provider(use_elevenlabs)
    stt_language = _stt_language(use_elevenlabs)
    stt_provider = _stt_provider_name()
    stt_model = stt_model_name()
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
        os.getenv("ELEVENLABS_TTS_LANGUAGE", "hi")
        if use_elevenlabs
        else sarvam_target_language_code
    )
    tts_voice = elevenlabs_voice_id if use_elevenlabs else sarvam_speaker

    logger.info(
        "Starting low-latency voice pipeline with stt=%s:%s llm=%s:%s tts_provider=%s tts=%s:%s voice=%s",
        f"{stt_provider}/{stt_model}",
        stt_language,
        llm_provider,
        llm_model,
        tts_provider,
        tts_model,
        tts_language,
        tts_voice,
    )

    # Phone callers enter their number on the keypad; the collector buffers those
    # DTMF digits so the agent can read them back instead of guessing from speech.
    dtmf_collector = DtmfCollector()

    # Caller memory and resume are keyed on Redis. If it's unreachable, every read
    # and write silently no-ops (so a live call never breaks) — which looks exactly
    # like "resume isn't working". Log one clear warning instead of failing quietly.
    redis_ok, redis_detail = redis_health_check()
    if not redis_ok:
        logger.warning(
            "Redis unreachable (%s) at %s — returning-caller memory and resume are "
            "DISABLED for this call. Start Redis or set REDIS_URL.",
            redis_detail,
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        )

    # Recognize returning callers: phone for telephony, IP for web sessions. The
    # lookup is safe when neither is available (-> None) and never raises.
    caller_ref = caller_ref_from_room(ctx.room)
    returning_caller = lookup_caller(caller_ref)
    if returning_caller is not None:
        logger.info(
            "Returning caller %s recognized (status=%s)",
            caller_ref,
            returning_caller.get("status"),
        )

    session = AgentSession(
        stt=build_stt(stt_language),
        llm=_build_llm(llm_model),
        tts=_build_tts(tts_model),
        turn_handling=_build_turn_handling_options(),
        vad=ctx.proc.userdata["vad"],
        use_tts_aligned_transcript=_env_bool("USE_TTS_ALIGNED_TRANSCRIPT", False),
        min_consecutive_speech_delay=_env_float(
            "MIN_CONSECUTIVE_SPEECH_DELAY", 0.05, min_value=0.0, max_value=2.0
        ),
        # AEC warmup disables interruptions on the agent's FIRST speaking turn so it
        # doesn't hear its own voice echo back (before the echo-canceller converges)
        # and self-interrupt — which chops the opening greeting. LiveKit's default is
        # 3.0s; 1.5s covers convergence on typical hardware while restoring barge-in
        # sooner. Too-low values (<0.5s) cause the greeting to stutter at startup.
        aec_warmup_duration=_env_float(
            "AEC_WARMUP_DURATION", 1.5, min_value=0.0, max_value=5.0
        ),
        user_away_timeout=None,
        userdata={
            "dtmf": dtmf_collector,
            "caller_ref": caller_ref,
            "returning_caller": returning_caller,
            "lead_progress": {},
            "lead_completed": False,
            "checkpoint_tasks": set(),
        },
    )

    register_dtmf_collector(ctx.room, dtmf_collector)

    async def persist_caller_memory():
        # Final safety net at call end. Background checkpoints already persist
        # progress mid-call, so first drain any in-flight checkpoint writes, then
        # upsert the latest snapshot. A completed lead already wrote a "completed"
        # record, so only partial progress is persisted here.
        pending = session.userdata.get("checkpoint_tasks")
        if pending:
            await asyncio.gather(*list(pending), return_exceptions=True)
        if not caller_ref or session.userdata.get("lead_completed"):
            return
        progress = session.userdata.get("lead_progress") or {}
        if not progress:
            return
        await asyncio.to_thread(
            upsert_caller_checkpoint,
            phone=caller_ref,
            status="partial",
            **progress,
        )

    ctx.add_shutdown_callback(persist_caller_memory)

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

    @session.on("user_interruption_detected")
    def _on_user_interruption_detected(ev):
        logger.debug(
            "User interruption detected: timestamp=%s probability=%s",
            getattr(ev, "timestamp", None),
            getattr(ev, "probability", None),
        )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev):
        logger.debug("False interruption detected; resuming agent speech")

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        logger.debug("User state changed: %s", getattr(ev, "new_state", None))

    # Cost accounting runs floating-point math and emits a log line on every usage
    # event, so it stays off the hot path by default. Enable COST_LOGGING_ENABLED
    # only when you need billing visibility.
    if _env_bool("COST_LOGGING_ENABLED", False):
        pricing = _pricing_config()
        last_logged_costs = {
            "stt": 0.0,
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

    if _noise_cancellation_enabled():
        auth_mode = (
            "ai-coustics license key" if _aicoustics_license_key() else "LiveKit Cloud"
        )
        logger.info(
            "Input noise cancellation ON (%s, model=%s) — STT receives denoised audio",
            auth_mode,
            _noise_cancellation_model().name,
        )
    else:
        logger.warning(
            "Input noise cancellation OFF — STT hears raw audio including background "
            "noise/voices. Set AICOUSTICS_LICENSE_KEY (self-hosted) or run on LiveKit "
            "Cloud, then ENABLE_NOISE_CANCELLATION=true."
        )

    await session.start(
        room=ctx.room,
        agent=WoiceVoiceAgent(),
        room_options=_room_options(),
    )

    if background_audio is not None:
        await background_audio.start(room=ctx.room, agent_session=session)
