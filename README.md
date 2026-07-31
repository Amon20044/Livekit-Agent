# StylMe LiveKit voice agent

This is the deliberately small LiveKit Cloud runtime for StylMe. Mongo-backed
configuration lives in `Go-Backend`; this worker loads a swarm snapshot for each
call and executes its validated DAG.

Runtime stack:

- OpenAI Responses API for agent reasoning
- Deepgram Nova-3 multilingual speech-to-text
- Sarvam Bulbul v3 text-to-speech
- Silero VAD and LiveKit's audio turn detector
- LiveKit `WarmTransferTask` for private briefing and room-based human transfer
- one Go control plane and one Mongo `calls` collection for both directions

Sarvam TTS uses the LiveKit plugin's built-in Bulbul v3 validation and defaults.
Agent Studio should expose only model/speaker combinations supported by the
installed plugin version.

There is no Redis memory, provider router, CRM, email, billing, or duplicated
workflow state in this package.

## Local run

The worker walks upward and reads only the repository root `.env`. Do not create
a second environment file in this directory.

```bash
uv sync
uv run -m livekit.agents download-files
uv run pytest
uv run ruff check src tests
uv run python src/agent.py dev
```

The Go API must be reachable at `LIVEKIT_CONTROL_PLANE_URL`; a local worker uses
`http://localhost:8081/v1`. Cloud deployment needs a public HTTPS Go service.
The worker reads the human destination from the active HIL node and the managed
outbound trunk from the selected swarm at runtime. Legacy workflow-level numbers
remain a temporary fallback for already-snapshotted calls. On a successful transfer, the human is moved into the
caller's original room and the AI session disconnects without deleting the room.

## LiveKit Cloud deployment

```bash
lk cloud auth
lk agent create --region ap-south
lk agent deploy
lk agent status
lk agent logs
```

`lk agent create` creates this project's `livekit.toml`. The old Samora/Woice
deployment identifier is intentionally not reused.

Required root environment keys are documented in the repository
`.env.example`: LiveKit, OpenAI, Deepgram, Sarvam, control-plane URL, and the
shared internal API key.

For Twilio SIP trunk and dispatch-rule setup, see `../VOICE_AI.md`.
