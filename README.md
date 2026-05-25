<div align="center">

# Woice AI

### Voice that doesn't miss a thing and actually finishes the conversation.

**Woice AI is a waitlist-stage voice automation platform for real business workflows: calls come in or go out, AI listens, understands intent, talks naturally, uses tools, saves the result, and escalates when needed.**

Website: [woice.vercel.app](https://woice.vercel.app)

Built on [LiveKit Agents](https://github.com/livekit/agents) with Deepgram, Gemini, Sarvam, ElevenLabs, Redis, and production-grade turn-taking.

</div>

---

## What this repo is

This project powers the Woice AI onboarding and pre-registration voice agent.

The public website explains the product vision. This agent does the live conversation: it welcomes a visitor, explains Woice in a natural way, learns the workflow they want to automate, confirms their contact details, sends a polished waitlist recap email, and saves the structured lead for follow-up.

Woice is currently in waitlist/pre-registration. The goal is not to promise instant access. The goal is to capture the right early users with the right context so the team can onboard them properly.

---

## Why Woice

Most voice AI demos feel magical for thirty seconds. Then the real phone call starts:

- The caller says `amon sharma 2000 at gmail dot com`, and the system writes garbage.
- The caller says `okay` while the agent is speaking, and the agent stops dead.
- A returning caller has to start from zero every time.
- The call ends, and the business gets an audio file instead of a completed workflow.

Woice AI is built around the boring, mission-critical parts that decide whether a voice agent is a toy or a business system:

- Reliable email and phone capture with readback confirmation.
- Natural multilingual calls across Hindi, English, and 50+ language workflows.
- Smart interruptions that ignore short backchannel like "okay" and "right".
- Returning-caller memory for phone-based recognition.
- Structured outcomes saved to the database.
- Follow-up email that feels premium and welcoming.
- A workflow layer that can connect to CRM, calendar, knowledge base, payments, support, messaging, and internal tools.

The value is not "AI can talk." The value is: **AI can talk and take action inside business systems.**

---

## What the waitlist agent does

On every call, the Woice AI concierge:

1. Greets the caller as Woice AI.
2. Explains the product as a voice workflow automation layer.
3. Asks what call workflow the user wants to automate.
4. Captures name, email, company, and workflow brief.
5. Reads back email and phone details before using them.
6. Confirms consent before saving or sending anything.
7. Sends a beautiful branded waitlist recap email.
8. Saves the structured lead in Redis.
9. Remembers phone callers for 30 days so they can resume later.

The workflow brief can include business type, inbound/outbound calls, current call volume, tools to integrate, urgency, and the exact outcome the user wants Woice to complete.

---

## Best use cases

Woice is strongest where calls are repetitive, high-volume, outcome-driven, and connected to existing software.

| Industry | Workflow Woice can complete |
|---|---|
| Clinics and hospitals | Book appointments, collect symptoms, send confirmations, update patient CRM |
| Real estate | Qualify property leads, answer inventory questions, book site visits, alert sales |
| E-commerce and logistics | Verify caller, check order status, create tickets, send updates |
| EdTech and coaching | Explain programs, qualify students, book demo classes, send brochures |
| Support teams | Intake issues, create tickets, attach summaries, escalate urgent cases |
| Sales teams | Handle missed calls, qualify leads, schedule meetings, update CRM |
| Finance and collections | Payment reminders, payment-link requests, callback scheduling |
| Recruitment | Screen applicants, capture availability, summarize fit, hand off to humans |

Bad use case: "Talk like a human for fun."

Good use case: "Qualify 500 real estate leads per day and book site visits automatically."

---

## Integrations Woice is built for

Woice is a 3rd-gen voice automation layer: telephony in, business outcome out.

| Layer | Examples |
|---|---|
| Telephony | Twilio, SIP, LiveKit SIP, Exotel, Plivo, Telnyx, Vonage |
| CRM | HubSpot, Salesforce, Zoho, Pipedrive, Freshsales, custom CRM |
| Calendar | Google Calendar, Outlook, Cal.com, Calendly, internal booking systems |
| Knowledge | Website pages, PDFs, Notion, Google Drive, docs, databases, help centers |
| Automation | Zapier, Make, n8n, Pipedream, webhooks |
| Support | Zendesk, Freshdesk, Intercom, Gorgias, ServiceNow |
| Messaging | WhatsApp, SMS, email, Slack, Discord, Telegram |
| Payments | Razorpay, Stripe, payment-link APIs |

Typical webhook events:

```json
{
  "event": "lead.qualified",
  "caller": "+919876543210",
  "name": "Rahul",
  "intent": "book_demo",
  "summary": "Interested in weekend Java backend course",
  "next_action": "send_payment_link"
}
```

---

## Architecture

```text
Caller
  |
  v
Twilio / SIP / LiveKit
  |
  v
LiveKit room
  |
  v
Deepgram Nova-3 STT
  |
  v
Silero VAD + LiveKit multilingual turn detector
  |
  v
Gemini / Bedrock / Groq LLM
  |
  v
Tool router
  |-- waitlist lead capture
  |-- DTMF keypad phone capture
  |-- live web search
  |-- Redis caller memory
  |-- email recap
  |
  v
Sarvam / ElevenLabs TTS
  |
  v
Structured lead + beautiful waitlist email
```

Typed input rides the same LiveKit `lk.chat` pipeline as speech, so a caller can type an email address and the agent handles it just like spoken input.

---

## Production conversation feel

The agent is tuned for a natural, marketable voice experience:

- Short spoken responses, usually one to three sentences.
- Hindi-first prompts with multilingual STT/TTS, so callers can switch languages
  mid-sentence without the agent announcing it.
- Email readback before commit.
- Confirm-before-save behavior.
- False-interruption recovery.
- Preemptive generation and TTS for fast first audio.
- Silero VAD with explicit production defaults.
- LiveKit multilingual turn detector.
- LiveKit Cloud adaptive interruption when available.
- Enhanced noise cancellation enabled automatically for `*.livekit.cloud` projects.

The practical target is: fast enough to feel alive, careful enough to not lose the lead.

---

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env.local
uv run python src/agent.py download-files
uv run python src/agent.py console
```

Run modes:

| Command | Use it for |
|---|---|
| `uv run python src/agent.py console` | Local terminal conversation |
| `uv run python src/agent.py dev` | Development worker for web/telephony |
| `uv run python src/agent.py start` | Production worker |

LiveKit credentials can be loaded with the LiveKit CLI:

```bash
lk cloud auth
lk app env -w -d .env.local
```

This repo expects the LiveKit CLI docs features from `lk` 2.15.0+. If your CLI is older, upgrade before using `lk docs`.

---

## Key configuration

```env
COMPANY_NAME=Woice AI
COMPANY_WEBSITE=https://woice.vercel.app
WOICE_WAITLIST_URL=https://woice.vercel.app
WOICE_REPLY_TO=

CALLER_MEMORY_TTL_SECONDS=2592000
LEAD_TTL_SECONDS=86400

DEEPGRAM_API_KEY=
DEEPGRAM_STT_MODEL=nova-3
DEEPGRAM_STT_LANGUAGE=multi
DEEPGRAM_ENDPOINTING_MS=25
DEEPGRAM_FILLER_WORDS=true
DEEPGRAM_SMART_FORMAT=false

GEMINI_LLM_MODEL=gemini-2.5-flash-lite
GEMINI_THINKING_BUDGET=0
GEMINI_MAX_OUTPUT_TOKENS=220

MIN_ENDPOINTING_DELAY=0.35
MAX_ENDPOINTING_DELAY=1.2
ENDPOINTING_MODE=dynamic

INTERRUPTION_MODE=vad
MIN_INTERRUPTION_DURATION=0.5
MIN_INTERRUPTION_WORDS=6
FALSE_INTERRUPTION_TIMEOUT=2.0

PREEMPTIVE_GENERATION=true
PREEMPTIVE_TTS=false
PREEMPTIVE_MAX_SPEECH_DURATION=2.5

VAD_MIN_SPEECH_DURATION=0.04
VAD_MIN_SILENCE_DURATION=0.42
VAD_PREFIX_PADDING_DURATION=0.45
VAD_ACTIVATION_THRESHOLD=0.52
VAD_SAMPLE_RATE=16000

ELEVENLABS_TTS_MODEL=eleven_flash_v2_5
ELEVENLABS_TTS_LANGUAGE=auto
ELEVENLABS_INACTIVITY_TIMEOUT=180
ELEVENLABS_TEXT_NORMALIZATION=auto
ELEVENLABS_SYNC_ALIGNMENT=true
```

Deepgram interim transcripts, Silero VAD, and the LiveKit multilingual turn
detector work together for end-of-turn timing and interruption handling.

`MIN_INTERRUPTION_WORDS=6` is the intelligent barge-in gate: VAD and STT keep
detecting while the caller speaks, but the agent keeps talking through up to five
words of backchannel ("okay", "haan right", "yeah sure got it") and only yields on
the sixth word, so a real interruption still cuts through. Lower it toward 2-3 for
snappier barge-in if the agent ever talks over genuine interruptions.

For LiveKit Cloud deployments, set `INTERRUPTION_MODE=adaptive` after your Cloud inference credentials are working. The code also defaults enhanced noise cancellation to on for `*.livekit.cloud` URLs.

### Per-number checkpoints

As the agent learns each detail it upserts a background checkpoint to the caller's
phone-keyed Redis record (`note_lead_progress` runs the write off the event loop, so
the conversation never blocks). Because the latest snapshot is persisted the moment
it's captured, a dropped call, a rate-limited LLM, or a crash still leaves a
resumable lead — and a returning caller is greeted with what was already collected.
A completed lead is never downgraded to partial. The call-shutdown hook drains any
in-flight checkpoint and writes a final snapshot as a safety net.

### Audio reliability

ElevenLabs `auto_mode` stays on so the plugin streams one sentence or phrase at a
time without a manual chunk schedule. The bundled default `mp3_22050_32` keeps
streaming socket degrades to a tiny silence rather than stalling the MP3 decoder —
the usual cause of choppy / "lost packet" audio mid-call. The streaming WebSocket is
also held open for up to `ELEVENLABS_INACTIVITY_TIMEOUT` seconds (180 max) so a quiet
stretch doesn't drop the connection and glitch the next reply. Note the
agent→caller leg is WebRTC/Opus, which already does packet-loss concealment;
enhanced noise cancellation (on by default for `*.livekit.cloud`) further protects
that path.

---

## Tests

Agent behavior is tested with `pytest`.

```bash
uv run pytest
uv run ruff format
uv run ruff check
```

The suite covers prompt expectations, turn-handling defaults, VAD loading, email/phone normalization, DTMF capture, Redis lead saving, caller memory, and the branded HTML waitlist email.

---

## Project structure

```text
src/
  agent.py                 # Worker entrypoint
  app/
    agent_session.py       # AgentSession wiring, VAD, noise, memory, metrics
    dtmf.py                # Keypad collector
  prompts/
    instructions.py        # Woice AI onboarding persona and waitlist flow
  tools/
    company.py             # Waitlist email, lead save, Redis caller memory
    telephony.py           # DTMF tool
  inferences/              # STT, LLM, TTS, turn detector builders
  audio/                   # Background audio
  telemetry/               # Cost accounting
tests/
  test_agent.py
  test_company_leads.py
  test_dtmf.py
  test_tools.py
```

---

## Roadmap

Woice today is the reliable voice layer and waitlist onboarding flow. The platform vision is bigger:

- Multi-tenant dashboard to configure agents without code.
- Live transcripts and lead inbox.
- Conversation analytics: conversion, drop-off, latency, language mix.
- Prompt and voice A/B testing.
- CRM, calendar, support, messaging, and payment integrations.
- Streaming RAG for company-specific answers.
- Human escalation and callback workflows.
- Usage-based billing from per-call telemetry.
- 3D avatar experiences for web onboarding and demos.

The runtime is the wedge. The platform is the product.

---

## License

MIT. See [LICENSE](LICENSE).
