import logging
import os
from pathlib import Path

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
    MetricsCollectedEvent,
    RoomInputOptions,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import silero, elevenlabs, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from welcome_agent import WelcomeAgent, HealthcareSessionData

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
elevenlabs_model = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
gemini_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(
        "Starting voice pipeline with stt=nova-3-general llm=%s tts=%s voice=%s",
        gemini_model,
        elevenlabs_model,
        elevenlabs_voice_id,
    )

    session = AgentSession[HealthcareSessionData](
        userdata=HealthcareSessionData(),
        stt=deepgram.STT(
            model="nova-3-general",
            language="en",
            api_key=deepgram_api_key,
        ),
        llm=google.LLM(
            model=gemini_model,
            api_key=google_api_key,
        ),
        tts=elevenlabs.TTS(
            model=elevenlabs_model,
            voice_id=elevenlabs_voice_id,
            api_key=elevenlabs_api_key,
            language="en",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=EnglishModel(),
            interruption={"mode": "vad"},
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        logger.info("Session ended")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        room=ctx.room,
        agent=WelcomeAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.6),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.6),
        ],
    )
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
