<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Agents Starter - Python

A complete starter project for building voice AI apps with [LiveKit Agents for Python](https://github.com/livekit/agents) and [LiveKit Cloud](https://cloud.livekit.io/).

The starter project includes:

- A simple voice AI assistant, ready for extension and customization
- A voice AI pipeline built on [LiveKit Inference](https://docs.livekit.io/agents/models/inference)
  with [models](https://docs.livekit.io/agents/models) from OpenAI, Cartesia, and Deepgram. More than 50 other model providers are supported, including [Realtime models](https://docs.livekit.io/agents/models/realtime)
- Eval suite based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/start/testing/)
- [LiveKit Turn Detector](https://docs.livekit.io/agents/logic/turns/turn-detector/) for contextually-aware speaker detection, with multilingual support
- [Background voice cancellation](https://docs.livekit.io/transport/media/noise-cancellation/)
- Deep session insights from LiveKit [Agent Observability](https://docs.livekit.io/deploy/observability/)
- A Dockerfile ready for [production deployment to LiveKit Cloud](https://docs.livekit.io/deploy/agents/)

This starter app is compatible with any [custom web/mobile frontend](https://docs.livekit.io/frontends/) or [telephony](https://docs.livekit.io/telephony/).

## Using coding agents

This project is designed to work with coding agents like [Claude Code](https://claude.com/product/claude-code), [Cursor](https://www.cursor.com/), and [Codex](https://openai.com/codex/).

For your convenience, LiveKit offers both a CLI and an [MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/) that can be used to browse and search its documentation. The [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) (`lk docs`) works with any coding agent that can run shell commands. Install it for your platform:

**macOS:**

```console
brew install livekit-cli
```

**Linux:**

```console
curl -sSL https://get.livekit.io/cli | bash
```

**Windows:**

```console
winget install LiveKit.LiveKitCLI
```

The `lk docs` subcommand requires version 2.15.0 or higher. Check your version with `lk --version` and update if needed. Once installed, your coding agent can search and browse LiveKit documentation directly from the terminal:

```console
lk docs search "voice agents"
lk docs get-page /agents/start/voice-ai-quickstart
```

See the [Using coding agents](https://docs.livekit.io/intro/coding-agents/) guide for more details, including MCP server setup.

The project includes a complete [AGENTS.md](AGENTS.md) file for these assistants. You can modify this file to suit your needs. To learn more about this file, see [https://agents.md](https://agents.md).

## Dev Setup

Create a project from this template with the LiveKit CLI (recommended):

```bash
lk cloud auth
lk agent init my-agent --template agent-starter-python
```

The CLI clones the template and configures your environment. Then follow the rest of this guide from [Run the agent](#run-the-agent).

<details>
<summary>Alternative: Manual setup without the CLI</summary>

Clone the repository and install dependencies to a virtual environment:

```console
cd agent-starter-python
uv sync
```

Sign up for [LiveKit Cloud](https://cloud.livekit.io/) then set up the environment by copying `.env.example` to `.env.local` and filling in the required keys:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

You can load the LiveKit environment automatically using the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/):

```bash
lk cloud auth
lk app env -w -d .env.local
```

</details>

## Run the agent

Before your first run, you must download certain models such as [Silero VAD](https://docs.livekit.io/agents/logic/turns/vad/) and the [LiveKit turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/):

```console
uv run python src/agent.py download-files
```

Next, run this command to speak to your agent directly in your terminal:

```console
uv run python src/agent.py console
```

To run the agent for use with a frontend or telephony, use the `dev` command:

```console
uv run python src/agent.py dev
```

In production, use the `start` command:

```console
uv run python src/agent.py start
```

## Frontend & Telephony

Get started quickly with our pre-built frontend starter apps, or add telephony support:

| Platform | Link | Description |
|----------|----------|-------------|
| **Web** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) | Web voice AI assistant with React & Next.js |
| **iOS/macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) | Native iOS, macOS, and visionOS voice AI assistant |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) | Cross-platform voice AI assistant app |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) | Native mobile app with React Native & Expo |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) | Native Android app with Kotlin & Jetpack Compose |
| **Web Embed** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) | Voice AI widget for any website |
| **Telephony** | [Documentation](https://docs.livekit.io/telephony/) | Add inbound or outbound calling to your agent |

For advanced customization, see the [complete frontend guide](https://docs.livekit.io/frontends/).

## Tests and evals

This project includes a complete suite of evals, based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/start/testing/). To run them, use `pytest`.

```console
uv run pytest
```

## Agent Optimization

This agent is tuned for a fast, lively voice experience: the user stops speaking, the system detects the turn boundary quickly, the LLM starts producing a short answer early, TTS begins streaming audio before the full answer is complete, and a quiet background track keeps the room from feeling sterile.

The pipeline is:

```text
microphone audio -> LiveKit room -> Deepgram multilingual STT -> turn handling -> Gemini LLM -> Sarvam TTS -> LiveKit room audio
                                                -> background ambience track
```

### Latency Budget

Voice latency is the sum of several small delays:

- **Endpointing latency**: how long the system waits after silence before deciding the user is done.
- **STT latency**: how quickly partial and final transcripts arrive.
- **LLM first-token latency**: how quickly Gemini starts producing the answer.
- **TTS first-audio latency**: how quickly Sarvam starts returning playable audio.
- **Network/media latency**: how fast LiveKit can route audio between the browser, server, and agent.

The current defaults optimize the “first useful audio” path rather than maximum reasoning depth. That is why the agent uses low endpointing delays, interim transcripts, preemptive generation, small TTS chunks, a short max answer length, and Gemini thinking budget `0`.

Study links:

- [LiveKit turn handling options](https://docs.livekit.io/reference/agents/turn-handling-options/)
- [LiveKit speech and background audio](https://docs.livekit.io/agents/build/audio/)
- [Deepgram STT integration for LiveKit Agents](https://docs.livekit.io/agents/integrations/deepgram/)
- [Sarvam TTS integration for LiveKit Agents](https://docs.livekit.io/agents/models/tts/sarvam/)
- [Google Gemini thinking controls](https://ai.google.dev/gemini-api/docs/thinking)

### Turn Detection and Endpointing

Turn detection answers: “Is the user still speaking, or should the agent answer now?”

This project combines LiveKit turn handling with Silero VAD and the multilingual turn detector. VAD detects speech versus silence from the audio signal. The turn detector adds linguistic context across supported languages, so the agent can be less naive than “silence means done.”

Important env values:

```env
MIN_ENDPOINTING_DELAY=0.22
MAX_ENDPOINTING_DELAY=0.9
ENDPOINTING_MODE=dynamic
ENDPOINTING_ALPHA=0.55
DEEPGRAM_ENDPOINTING_MS=25
```

Conceptually:

- Lower `MIN_ENDPOINTING_DELAY` means the agent can react sooner after a pause.
- Lower `MAX_ENDPOINTING_DELAY` caps how long the agent waits when the turn detector is uncertain.
- `dynamic` endpointing lets LiveKit adapt within the min/max range based on pause behavior.
- Deepgram endpointing and interim results help transcripts arrive while the user is still speaking.

ML concept to study: voice activity detection and endpointing are sequence classification problems over streaming audio. The model is estimating whether recent frames represent speech, silence, or a likely turn boundary.

Study links:

- [LiveKit VAD](https://docs.livekit.io/agents/logic/turns/vad/)
- [LiveKit turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/)
- [Deepgram endpointing and interim results](https://developers.deepgram.com/docs/understand-endpointing-interim-results)

### STT Speed

The agent uses Deepgram with:

```env
DEEPGRAM_STT_MODEL=nova-3-general
DEEPGRAM_STT_LANGUAGE=multi
DEEPGRAM_ENDPOINTING_MS=25
DEEPGRAM_SMART_FORMAT=false
DEEPGRAM_FILLER_WORDS=false
```

In code, STT is configured with `interim_results=True` and `no_delay=True`. Interim results let downstream logic begin earlier. `no_delay` asks the provider not to hold transcripts for extra context. Smart formatting and filler words are disabled by default because they can add processing work and are less important for a spoken news assistant than quick intent capture.

ML concept to study: streaming ASR is a partial decoding problem. The recognizer emits provisional hypotheses before finalizing the utterance. Faster partial hypotheses can reduce latency, but late words may be corrected as more audio context arrives.

Study link:

- [Deepgram STT integration for LiveKit Agents](https://docs.livekit.io/agents/integrations/deepgram/)

### LLM Reasoning Budget

The agent is configured for quick spoken answers:

```env
GEMINI_LLM_MODEL=gemini-2.5-flash-lite
GEMINI_THINKING_BUDGET=0
GEMINI_MAX_OUTPUT_TOKENS=220
GEMINI_TEMPERATURE=0.35
```

For voice UX, the best answer is often not the longest or deepest answer. The user is waiting in real time, so the agent should answer briefly and ask follow-up questions when needed. `GEMINI_THINKING_BUDGET=0` disables extra thinking tokens for Gemini 2.5 style models, reducing inference-time reasoning overhead. `GEMINI_MAX_OUTPUT_TOKENS=220` keeps replies compact so TTS can start and finish sooner.

ML concept to study: inference-time reasoning trades latency and cost for deeper search. A higher thinking budget can improve hard reasoning, but it increases time-to-first-token. For conversational voice, low or zero budget is usually better unless the task is complex.

Study links:

- [Google Gemini thinking controls](https://ai.google.dev/gemini-api/docs/thinking)
- [Vertex AI thinking budget guidance](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking)

### Preemptive Generation

The agent enables:

```env
PREEMPTIVE_GENERATION=true
PREEMPTIVE_TTS=true
PREEMPTIVE_MAX_SPEECH_DURATION=2.5
PREEMPTIVE_MAX_RETRIES=1
```

Preemptive generation lets the agent begin preparing a response before the user turn is fully finalized. Preemptive TTS can begin preparing audio from early LLM output. This reduces perceived latency because work overlaps instead of happening strictly one step after another.

The tradeoff: if the user continues speaking or the partial transcript changes, preemptive work may be discarded. This is why the generated speech duration and retries are kept bounded.

ML/system concept to study: speculative execution. The system starts likely work early based on partial evidence, then either commits it or discards it when better evidence arrives.

Study link:

- [LiveKit turn handling options](https://docs.livekit.io/reference/agents/turn-handling-options/)

### TTS Streaming

The agent uses Sarvam Bulbul with streaming-focused options:

```env
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TARGET_LANGUAGE_CODE=hi-IN
SARVAM_SPEAKER=shubh
SARVAM_PACE=1.0
SARVAM_TEMPERATURE=0.6
SARVAM_MIN_BUFFER_SIZE=50
SARVAM_MAX_CHUNK_LENGTH=150
```

TTS latency is heavily affected by how much text the provider waits for before generating audio. Smaller chunks usually reduce time-to-first-audio, while larger chunks can improve prosody and naturalness. Sarvam exposes buffering and chunk length controls, so tune `SARVAM_MIN_BUFFER_SIZE` and `SARVAM_MAX_CHUNK_LENGTH` if the voice feels too eager or too delayed.

`USE_TTS_ALIGNED_TRANSCRIPT=false` avoids alignment work that is useful for captions but not essential for fastest spoken output.

ML concept to study: neural TTS performs sequence-to-sequence generation from text tokens to acoustic frames. Chunking controls how much future text context the model sees before it starts producing audio, so it directly affects latency versus prosody.

Study links:

- [LiveKit Sarvam TTS plugin](https://docs.livekit.io/agents/models/tts/sarvam/)
- [Sarvam docs](https://docs.sarvam.ai/)

### Background Ambience

The agent adds a separate low-volume background audio track:

```env
BACKGROUND_AUDIO_ENABLED=true
BACKGROUND_AMBIENT_CLIP=OFFICE_AMBIENCE
BACKGROUND_AMBIENT_VOLUME=0.18
BACKGROUND_THINKING_SOUND_ENABLED=true
BACKGROUND_THINKING_VOLUME=0.16
BACKGROUND_THINKING_VOLUME_ALT=0.12
BACKGROUND_AUDIO_STREAM_TIMEOUT_MS=200
```

This does not come from the TTS voice. LiveKit publishes ambience as its own room audio track using `BackgroundAudioPlayer`. That matters because the agent can keep its speech clean while still making the room feel alive. Thinking sounds are synchronized with the agent lifecycle, so quiet keyboard sounds can play while the agent is working.

Keep ambience subtle. If users strain to hear speech, lower `BACKGROUND_AMBIENT_VOLUME` to `0.08` or disable thinking sounds.

Study links:

- [LiveKit background audio docs](https://docs.livekit.io/agents/build/audio/#adding-background-audio)
- [Python BackgroundAudioPlayer reference](https://docs.livekit.io/reference/python/livekit/agents/voice/background_audio.html)

### Noise Cancellation

The fast profile defaults to:

```env
ENABLE_NOISE_CANCELLATION=false
```

Noise cancellation can improve audio quality in noisy rooms, but it is another processing stage. For lowest latency in a controlled environment, keeping it off is reasonable. If users are in noisy places, turn it back on and measure the tradeoff.

Study link:

- [LiveKit noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation/)

### Tuning Guide

Use these presets when adjusting production `ENV_LOCAL`:

**Fastest**

```env
MIN_ENDPOINTING_DELAY=0.15
MAX_ENDPOINTING_DELAY=0.65
GEMINI_THINKING_BUDGET=0
GEMINI_MAX_OUTPUT_TOKENS=160
SARVAM_MIN_BUFFER_SIZE=30
SARVAM_MAX_CHUNK_LENGTH=100
BACKGROUND_AMBIENT_VOLUME=0.10
```

Best when responsiveness matters more than perfectly polished speech.

**Balanced**

```env
MIN_ENDPOINTING_DELAY=0.22
MAX_ENDPOINTING_DELAY=0.9
GEMINI_THINKING_BUDGET=0
GEMINI_MAX_OUTPUT_TOKENS=220
SARVAM_MIN_BUFFER_SIZE=50
SARVAM_MAX_CHUNK_LENGTH=150
BACKGROUND_AMBIENT_VOLUME=0.18
```

This is the current default.

**More Thoughtful**

```env
MIN_ENDPOINTING_DELAY=0.35
MAX_ENDPOINTING_DELAY=1.4
GEMINI_THINKING_BUDGET=512
GEMINI_MAX_OUTPUT_TOKENS=320
SARVAM_MIN_BUFFER_SIZE=80
SARVAM_MAX_CHUNK_LENGTH=220
BACKGROUND_AMBIENT_VOLUME=0.12
```

Best for complex questions where a slightly slower answer is acceptable.

### Measuring Improvements

Use logs and metrics to separate the bottlenecks:

- If the delay is before transcript finalization, tune Deepgram endpointing and LiveKit endpointing.
- If transcript finalizes quickly but speech starts late, tune Gemini thinking/output tokens or Sarvam buffering.
- If first response after idle is slow, look for cold starts, model downloads, or EC2 CPU/memory pressure.
- If audio connects slowly or drops, inspect LiveKit server logs, ICE/TURN connectivity, and EC2 security groups.

The agent logs LiveKit metrics through `metrics.log_metrics`, so use those timing events before guessing. A good voice agent is optimized by measuring each stage, not by turning every knob to the minimum.

## Using this template repo for your own project

Once you've started your own project based on this repo, you should:

1. **Check in your `uv.lock`**: This file is currently untracked for the template, but you should commit it to your repository for reproducible builds and proper configuration management. (The same applies to `livekit.toml`, if you run your agents in LiveKit Cloud)

2. **Remove the git tracking test**: Delete the "Check files not tracked in git" step from `.github/workflows/tests.yml` since you'll now want this file to be tracked. These are just there for development purposes in the template repo itself.

3. **Add your own repository secrets**: You must [add secrets](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/using-secrets-in-github-actions) for `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` so that the tests can run in CI.

## Deploying to production

This project is production-ready and includes a working `Dockerfile`. To deploy it to LiveKit Cloud or another environment, see the [deploying to production](https://docs.livekit.io/deploy/agents/) guide.

### Deploy to EC2 with GitHub Actions (manual)

This repository includes a minimal manual workflow at `.github/workflows/deploy-ec2.yml` that deploys over SSH and starts the agent worker on your EC2 host.

Before using it:

- Ensure SSH access to the EC2 host from GitHub Actions runners.
- Add these GitHub repository secrets:
  - `EC2_HOST` (public IP or DNS)
  - `EC2_USER` (for example `ec2-user`)
  - `EC2_PRIVATE_KEY` (contents of your `.pem` key)
  - `ENV_LOCAL` (the full contents of your production `.env.local` file)

The workflow writes `ENV_LOCAL` to `/etc/my-agent.env` and `/opt/my-agent/.env.local` on EC2. It also derives `/opt/livekit/egress.yaml`, `/opt/livekit/ingress.yaml`, and the LiveKit server key block from the same `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` values.

How to run:

1. Open **Actions** and run **Deploy Agent To EC2**.
2. Optionally adjust `ref`, `app_dir`, `env_file`, and `ssh_port`.

The workflow uploads the repo and executes `scripts/ec2/start_agent_worker.sh` over SSH, then prints recent worker logs.

### Deploy from your machine with host/user/pem

You can also deploy directly without GitHub Actions:

```bash
scripts/ec2/deploy_via_ssh.sh <ec2_host> <ec2_user> <pem_file>
```

Optional arguments:

```bash
scripts/ec2/deploy_via_ssh.sh <ec2_host> <ec2_user> <pem_file> [app_dir] [env_file] [ssh_port]
```

## Self-hosted LiveKit

You can also self-host LiveKit instead of using LiveKit Cloud. See the [self-hosting](https://docs.livekit.io/transport/self-hosting/local/) guide for more information. If you choose to self-host, you'll need to also use [model plugins](https://docs.livekit.io/agents/models/#plugins) instead of LiveKit Inference and will need to remove the [LiveKit Cloud noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation/) plugin.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
