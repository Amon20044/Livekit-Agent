from livekit import agents

from app.agent_session import entrypoint, prewarm
from core.logging import _configure_log_levels
from settings import AGENT_NAME, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL

_configure_log_levels()


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
