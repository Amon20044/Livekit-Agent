import logging

from livekit import agents

from app.agent_session import entrypoint, prewarm
from settings import (
    AGENT_NAME,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_URL,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

server = agents.AgentServer(
    setup_fnc=prewarm,
    api_key=LIVEKIT_API_KEY,
    api_secret=LIVEKIT_API_SECRET,
    ws_url=LIVEKIT_URL,
)
server.rtc_session(agent_name=AGENT_NAME)(entrypoint)


if __name__ == "__main__":
    agents.cli.run_app(server)
