<div align="center">

# 🎙️ Woice

### Voice AI agents that actually finish the conversation.

**Real-time, multilingual voice agents that capture the details, read them back, remember the caller, and turn every call into structured, actionable data.**

Built on [LiveKit Agents](https://github.com/livekit/agents) · Powered by Deepgram, Gemini, Sarvam & ElevenLabs · Designed for India 🇮🇳 and the world.

</div>

---

## Why Woice

Voice AI demos look magical for thirty seconds. Then you put one on a real phone line and it falls apart on the boring, mission-critical 20%:

- It hears **"amon sharma 2000 at gmail dot com"** and writes down garbage.
- You say **"okay"** while it's talking and it stops dead, thinking you interrupted.
- A returning caller has to **start from zero** every single time.
- The call ends and you're left with **an audio file, not a lead.**

Those are the moments that decide whether a voice agent is a toy or a business. **Woice is built around getting those moments right** — accurate data capture, human-like turn-taking, multilingual conversation, and memory across calls — and then turning every call into clean, structured data you can act on.

The included reference agent is **DreamLaunch Studio**, a voice concierge that qualifies inbound leads, captures their brief, confirms it, and emails a recap with a booking link — fully in Hindi or English. It's a working blueprint for any "answer, qualify, capture, book" voice workflow: agencies, clinics, real estate, support desks, and beyond.

---

## ✨ What happens on a call

These are the caller-facing features — what someone actually experiences when they talk to a Woice agent.

| Feature | What the caller feels |
|---|---|
| 🗣️ **Natural multilingual conversation** | Speaks Hindi by default and switches to the caller's language mid-sentence — no robotic "please select a language." |
| ⚡ **Sub-second responsiveness** | Tuned end-to-end for "first useful audio" — interim transcripts, preemptive generation, and streaming TTS mean replies start almost immediately. |
| 🤫 **Smart interruption handling** | Single-word backchannel like *"okay," "right," "uh-huh"* no longer cuts the agent off — interrupting takes a real, multi-word utterance. Upgradable to LiveKit Cloud's adaptive barge-in model. |
| 🧠 **Remembers returning callers** | On phone calls, a caller is recognized by number — greeted by name if they finished before, or offered to pick up where they left off if they didn't. |
| 📧 **Reliable email & phone capture** | Callers can **type** their email/phone into the chat, **say** it slowly, or **enter it on the keypad** — and the agent always **reads it back to confirm.** |
| 🔢 **Keypad (DTMF) entry** | On phone calls, callers punch in their number on the dial pad and press `#`. No more spelling out ten digits over a noisy line. |
| ✅ **Confirm-before-commit** | The agent summarizes everything it captured and asks permission before sending anything. Malformed emails are caught and re-collected, never silently sent. |
| 🔎 **Live web knowledge** | Can pull current information mid-conversation via real-time search (Google News / AI Mode) when the caller asks about something recent. |
| 🎧 **A room that feels alive** | Subtle background ambience and quiet "thinking" sounds keep the call from feeling sterile or dropped. |
| 📨 **A real outcome** | Ends with an emailed recap + booking link, and a structured lead saved to the database — not just a transcript. |

---

## 🧰 What the platform gives builders

The features under the hood that make Woice agents reliable and extensible.

- **🔌 Swappable model stack.** LLM, STT, and TTS are all pluggable via environment variables — no code changes to switch providers.
  - **STT:** Deepgram `nova-3` (multilingual, smart-formatted for clean emails/numbers)
  - **LLM:** Google Gemini 2.5 Flash Lite by default, with AWS Bedrock and Groq as drop-in alternatives, plus an automatic fallback adapter
  - **TTS:** Sarvam Bulbul (Indian languages) or ElevenLabs (English), selectable per deployment
- **🧠 Adaptive turn-taking.** Silero VAD + LiveKit's multilingual turn detector + adaptive interruption, all tunable.
- **🛠️ Function-tool framework.** Tools are plain Python functions — lead capture + email, DTMF retrieval, and live search ship in the box; add your own in minutes.
- **🗃️ Stateful by design.** Redis-backed lead storage with TTLs, ready to grow into full cross-call memory (see [Roadmap](#-roadmap)).
- **💰 Cost & usage telemetry.** Optional per-turn and per-call cost accounting across STT/LLM/TTS, plus LiveKit observability.
- **🧪 Test-driven.** A full `pytest` suite covers instructions, tools, turn-handling, capture/validation, and DTMF — agent behavior is verified, not guessed.
- **🚀 Deploy anywhere.** Production `Dockerfile`, LiveKit Cloud support, and one-command EC2 deploy scripts + GitHub Actions.

---

## 🏗️ How it works

```text
 caller (web / phone / SIP)
        │  audio
        ▼
 ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐
 │ LiveKit room │──▶│ Deepgram STT     │──▶│ Turn handling│──▶│ Gemini LLM   │
 │ (WebRTC/SIP) │   │ (multilingual,   │   │ VAD + turn   │   │ + tools      │
 │              │   │  smart-format)   │   │ detector +   │   │ (lead, DTMF, │
 │              │◀──│                  │◀──│ adaptive     │◀──│  search...)  │
 └──────┬───────┘   └──────────────────┘   │ interruption │   └──────┬───────┘
        │  audio out                        └──────────────┘          │
        ▼                                                             ▼
   Sarvam / ElevenLabs TTS  ◀───────────────────────────────  Redis (lead state)
        +                                                             │
   background ambience track                                          ▼
                                                          Email recap + booking link
```

Text input rides the same pipeline: anything the caller **types** arrives on LiveKit's `lk.chat` stream and is handled exactly like speech — which is why typing an email "just works" alongside talking.

---

## 🚀 Quickstart

This project uses the [`uv`](https://docs.astral.sh/uv/) package manager.

```bash
# 1. Install dependencies
uv sync

# 2. Configure your environment
#    Copy the example and fill in your keys (LiveKit, Deepgram, Gemini, Sarvam/ElevenLabs, Redis, SMTP)
cp .env.example .env.local

# 3. Download required models (Silero VAD + turn detector) — first run only
uv run python src/agent.py download-files

# 4. Talk to the agent right in your terminal
uv run python src/agent.py console
```

Run modes:

| Command | Use it for |
|---|---|
| `uv run python src/agent.py console` | Talk to the agent locally in your terminal |
| `uv run python src/agent.py dev` | Run against a real frontend or telephony (dev) |
| `uv run python src/agent.py start` | Production worker |

> **Tip:** You can load your LiveKit credentials automatically with the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/): `lk cloud auth && lk app env -w -d .env.local`.

### Frontends & telephony

Woice works with any [custom web/mobile frontend](https://docs.livekit.io/frontends/) or [telephony](https://docs.livekit.io/telephony/) setup. The fastest start is the React app — and because it has a chat box, the **type-your-email** flow works out of the box:

| Platform | Starter |
|---|---|
| **Web (React/Next.js)** | [`agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) |
| **iOS / macOS** | [`agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) |
| **Flutter** | [`agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) |
| **React Native** | [`voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) |
| **Android** | [`agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) |
| **Web Embed** | [`agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) |
| **Telephony (SIP)** | [Docs](https://docs.livekit.io/telephony/) — enables inbound calls + DTMF keypad capture |

---

## 🧪 Tests

Agent behavior is covered by a `pytest` suite. **When you change instructions, tools, or turn-handling, write the test first** (see [AGENTS.md](AGENTS.md)).

```bash
uv run pytest          # run everything
uv run ruff format     # format
uv run ruff check      # lint
```

---

## ⚙️ Configuration & tuning

Woice is tuned for a fast, lively voice experience and is controlled almost entirely through environment variables in `.env.local`. The headline knobs:

### Conversation feel

```env
# Turn-taking & interruptions
INTERRUPTION_MODE=vad             # vad | adaptive  — adaptive barge-in needs LiveKit Cloud
MIN_INTERRUPTION_WORDS=2          # words required to interrupt; filters "okay"/"right" backchannel
MIN_ENDPOINTING_DELAY=0.22        # how soon the agent may answer after a pause
MAX_ENDPOINTING_DELAY=0.9         # cap on how long it waits when unsure
ENDPOINTING_MODE=dynamic

# Brand (one codebase, many studios) + caller memory
COMPANY_NAME=DreamLaunch Studio
COMPANY_WEBSITE=https://dreamlaunch.studio
CALLER_MEMORY_TTL_SECONDS=2592000  # how long a returning caller is remembered (30 days)

# Speech recognition
DEEPGRAM_STT_MODEL=nova-3-general
DEEPGRAM_STT_LANGUAGE=multi
DEEPGRAM_SMART_FORMAT=true         # formats spoken emails & numbers — keep on for capture accuracy

# Reasoning budget (kept low for snappy spoken replies)
GEMINI_LLM_MODEL=gemini-2.5-flash-lite
GEMINI_THINKING_BUDGET=0
GEMINI_MAX_OUTPUT_TOKENS=220
```

> **Interruption modes:** the default `vad` works everywhere and filters single-word backchannel via `MIN_INTERRUPTION_WORDS`. The smarter `adaptive` barge-in model runs on **LiveKit Cloud** inference (needs aligned-transcript STT — Deepgram qualifies). On a self-hosted server it returns `401` and falls back to VAD after a few noisy retries, so keep `vad` unless you deploy on LiveKit Cloud.

### Latency presets

Drop one of these into `.env.local` depending on your priority:

<details>
<summary><b>Fastest</b> — responsiveness over polish</summary>

```env
MIN_ENDPOINTING_DELAY=0.15
MAX_ENDPOINTING_DELAY=0.65
GEMINI_MAX_OUTPUT_TOKENS=160
SARVAM_MIN_BUFFER_SIZE=30
SARVAM_MAX_CHUNK_LENGTH=100
BACKGROUND_AMBIENT_VOLUME=0.10
```
</details>

<details open>
<summary><b>Balanced</b> — the current default</summary>

```env
MIN_ENDPOINTING_DELAY=0.22
MAX_ENDPOINTING_DELAY=0.9
GEMINI_MAX_OUTPUT_TOKENS=220
SARVAM_MIN_BUFFER_SIZE=50
SARVAM_MAX_CHUNK_LENGTH=150
BACKGROUND_AMBIENT_VOLUME=0.18
```
</details>

<details>
<summary><b>More thoughtful</b> — depth over speed</summary>

```env
MIN_ENDPOINTING_DELAY=0.35
MAX_ENDPOINTING_DELAY=1.4
GEMINI_THINKING_BUDGET=512
GEMINI_MAX_OUTPUT_TOKENS=320
SARVAM_MIN_BUFFER_SIZE=80
SARVAM_MAX_CHUNK_LENGTH=220
```
</details>

### The latency budget (and how to debug it)

Perceived delay is a sum of stages — optimize the one that's actually slow:

| Stage | What it is | Lever |
|---|---|---|
| **Endpointing** | Waiting after silence to decide the caller is done | `MIN/MAX_ENDPOINTING_DELAY`, Deepgram endpointing |
| **STT** | Time to partial/final transcript | interim results + `no_delay` (on by default) |
| **LLM** | Time to first token | `GEMINI_THINKING_BUDGET`, `GEMINI_MAX_OUTPUT_TOKENS` |
| **TTS** | Time to first audio | `SARVAM_MIN_BUFFER_SIZE`, `SARVAM_MAX_CHUNK_LENGTH` |
| **Network/media** | Routing audio over WebRTC/SIP | LiveKit server, ICE/TURN, host placement |

The agent emits LiveKit metrics via `metrics.log_metrics` — **measure each stage before turning knobs.** Helpful references: [turn handling](https://docs.livekit.io/reference/agents/turn-handling-options/) · [adaptive interruption](https://docs.livekit.io/agents/build/turns/interruptions/) · [Deepgram STT](https://docs.livekit.io/agents/integrations/deepgram/) · [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking) · [Sarvam TTS](https://docs.livekit.io/agents/models/tts/sarvam/) · [background audio](https://docs.livekit.io/agents/build/audio/).

---

## 📁 Project structure

```text
src/
├── agent.py                 # Worker entrypoint (keep this name — the Dockerfile uses it)
├── app/
│   ├── agent_session.py     # AnchorVoiceAgent + session wiring (STT/LLM/TTS/turn/DTMF)
│   └── dtmf.py              # DTMF keypad collector for SIP callers
├── inferences/             # Model builders: stt, llm, tts, turn detection, voice selection
├── prompts/instructions.py  # Agent persona + capture/confirmation playbook
├── tools/                   # Function tools: lead capture+email, DTMF, web search
├── audio/background.py      # Background ambience + thinking sounds
├── telemetry/costs.py       # Per-turn/per-call cost accounting
└── core/                    # Env helpers + logging
tests/                       # pytest suite (TDD for all behavior changes)
```

---

## 📦 Deploying to production

A production-ready `Dockerfile` is included. Deploy to [LiveKit Cloud](https://docs.livekit.io/deploy/agents/) or your own infrastructure.

### EC2 (GitHub Actions)

Run **Deploy Agent To EC2** from the **Actions** tab. It packages the app, ships it over SSH, builds the image, and starts the worker container. Required repository secrets:

| Secret | Purpose |
|---|---|
| `EC2_HOST` | Public IP or DNS of the host |
| `EC2_USER` | SSH user (e.g. `ec2-user`) |
| `EC2_PRIVATE_KEY` | Contents of your `.pem` key |
| `ENV_LOCAL` | Full contents of your production `.env.local` |

The workflow installs `ENV_LOCAL` to `/etc/my-agent.env` and `/opt/my-agent/.env.local`, then builds and runs the agent container. It **does not touch your LiveKit server, SIP, ingress, or egress configuration** — those are managed independently.

### EC2 (from your machine)

```bash
scripts/ec2/deploy_via_ssh.sh <ec2_host> <ec2_user> <pem_file> [app_dir] [env_file] [ssh_port]
```

### Self-hosting LiveKit

You can self-host the LiveKit server instead of using LiveKit Cloud — see the [self-hosting guide](https://docs.livekit.io/transport/self-hosting/local/). Note that some Cloud-only features (adaptive interruption inference, Cloud noise cancellation) gracefully degrade or need alternatives when self-hosted.

---

## 🗺️ Roadmap

Woice today is the **agent runtime** — the hard, reliable voice layer. The vision is the **platform around it**: turning conversations into a product surface you can sell.

> The items below are planned/in-progress, not yet shipped.

- **🧠 Returning-caller memory.** Recognize a caller by phone number on connect, and resume where they left off. New caller → fresh intake. Partial history → *"Welcome back — let's pick up where we stopped."* Completed before → greet them by name and ask if there's anything else. All state persisted in Redis as the source of truth.
- **📊 Multi-tenant dashboard.** A control plane to spin up and configure agents (persona, voice, language, tools) without touching code, watch **live transcripts**, and browse every captured lead.
- **📈 Conversation analytics.** Conversion rates, drop-off points, capture success, latency percentiles, and language mix — per agent and per campaign.
- **🔁 Prompt & voice A/B testing.** Compare personas, voices, and latency presets on live traffic and keep what converts.
- **🔗 CRM & calendar integrations.** Push qualified leads straight into HubSpot/Salesforce/Sheets and book real calendar slots.
- **💳 Usage-based billing.** Built on the existing cost telemetry, so every account is metered and monetizable.

The thesis: the runtime is the wedge; **the dashboard, the data, and the integrations are the platform.**

---

## 🤝 Working with coding agents

This repo is built to be developed with coding agents (Claude Code, Cursor, Codex). See [AGENTS.md](AGENTS.md) for project conventions — most importantly, **test-driven development for any agent-behavior change**, and how to use the [LiveKit docs CLI/MCP](https://docs.livekit.io/reference/developer-tools/docs-mcp/) for up-to-date references.

## 📄 License

MIT — see [LICENSE](LICENSE).
