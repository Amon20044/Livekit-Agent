import logging
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    TurnHandlingOptions,
    metrics,
)
from livekit.plugins import deepgram, elevenlabs, google, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import ai_coustics

from tools import search_ai_mode, search_latest_news

logger = logging.getLogger("agent")

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
gemini_model = os.getenv("GEMINI_LLM_MODEL", "gemini-3-flash-preview")
gemini_thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low")


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
- Before searching, the tool will tell the user: Yes, sure, let me search that for you.
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
        "Starting voice pipeline with stt=%s llm=%s tts=%s voice=%s",
        stt_model,
        llm_model,
        tts_model,
        elevenlabs_voice_id,
    )

    llm_kwargs = {
        "model": llm_model,
        "api_key": google_api_key,
    }
    if llm_model.startswith("gemini-3"):
        llm_kwargs["thinking_config"] = {"thinking_level": gemini_thinking_level}

    session = AgentSession(
        stt=deepgram.STT(
            model=stt_model,
            language=deepgram_language,
            api_key=deepgram_api_key,
        ),
        llm=google.LLM(**llm_kwargs),
        tts=elevenlabs.TTS(
            model=tts_model,
            voice_id=elevenlabs_voice_id,
            api_key=elevenlabs_api_key,
            language=elevenlabs_language,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=EnglishModel(),
            interruption={"mode": "vad"},
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=_env_bool("PREEMPTIVE_GENERATION", True),
        use_tts_aligned_transcript=True,
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        logger.info("Session ended")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        room=ctx.room,
        agent=AnchorVoiceAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_L),
        ),
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
