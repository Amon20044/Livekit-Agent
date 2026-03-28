import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc

# BGM
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    cli,
    room_io,
    TurnHandlingOptions
)
from livekit.plugins import deepgram, elevenlabs, google, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

ENV_PATH = Path(__file__).resolve().parents[1] / ".env.local"
load_dotenv(ENV_PATH)


def _env(name: str, default: str | None = None) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    cleaned = raw_value.strip().strip('"').strip("'")
    if " #" in cleaned:
        cleaned = cleaned.split(" #", 1)[0].strip()

    return cleaned or default


def _env_bool(name: str, default: bool) -> bool:
    raw_value = _env(name)
    if raw_value is None:
        return default

    return raw_value.lower() in {"1", "true", "yes", "on"}


AGENT_NAME = _env("LIVEKIT_AGENT_NAME", "my-agent") or "my-agent"
DEEPGRAM_STT_MODEL = (_env("DEEPGRAM_STT_MODEL", "nova-3-general") or "").split("/")[-1] or "nova-3-general"
DEEPGRAM_STT_LANGUAGE = _env("DEEPGRAM_STT_LANGUAGE", "en")
GEMINI_LLM_MODEL = (_env("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite") or "").split("/")[-1] or "gemini-2.5-flash-lite"
ELEVENLABS_TTS_MODEL = (
    (_env("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2") or "").split("/")[-1]
    or "eleven_multilingual_v2"
)
ELEVENLABS_TTS_LANGUAGE = _env("ELEVENLABS_TTS_LANGUAGE", "en")
ELEVENLABS_VOICE_ID = _env("ELEVENLABS_VOICE_ID")
PREEMPTIVE_GENERATION = _env_bool("PREEMPTIVE_GENERATION", True)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(
        "Starting voice pipeline with stt=%s llm=%s tts=%s voice=%s preemptive=%s",
        DEEPGRAM_STT_MODEL,
        GEMINI_LLM_MODEL,
        ELEVENLABS_TTS_MODEL,
        ELEVENLABS_VOICE_ID,
        PREEMPTIVE_GENERATION,
    )

    # Use the same provider mix as the working agent-worker setup while
    # staying on LiveKit Inference so we don't need lockfile changes here.
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand.
        stt=deepgram.STT(
            model=DEEPGRAM_STT_MODEL,
            language=DEEPGRAM_STT_LANGUAGE,
        ),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response.
        llm=google.LLM(model=GEMINI_LLM_MODEL),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear.
        tts=elevenlabs.TTS(
            model=ELEVENLABS_TTS_MODEL,
            voice_id=ELEVENLABS_VOICE_ID,
            language=ELEVENLABS_TTS_LANGUAGE,
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond.
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption={"mode": "vad"},
        ),
        vad=ctx.proc.userdata["vad"],
        # Allow the LLM to generate a response while waiting for the end of turn.
        preemptive_generation=PREEMPTIVE_GENERATION,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.6),
        # # play keyboard typing sound when the agent is thinking, here you can also use your own custom thinking sound by providing a list of file paths to different audio clips for variety
        # thinking_sound=[
        #     AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
        #     AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.6),
        # ],
    )

    await background_audio.start(room=ctx.room, agent_session=session)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
