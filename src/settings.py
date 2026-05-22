import os

from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_FILE_PATH = os.path.join(ROOT_DIR, ".env.local")
load_dotenv(ENV_FILE_PATH)

# LiveKit connection
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "woice-ai-agent")

# Brand identity. Configurable so one codebase can serve many deployments; defaults
# keep the bundled Woice AI waitlist concierge working out of the box.
COMPANY_NAME = os.getenv("COMPANY_NAME", "Woice AI")
COMPANY_WEBSITE = os.getenv("COMPANY_WEBSITE", "https://woice.vercel.app")

# Provider API keys
speechmatics_api_key = os.getenv("SPEECHMATICS_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
sarvam_api_key = os.getenv("SARVAM_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Model config
elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
sarvam_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
sarvam_target_language_code = os.getenv("SARVAM_TARGET_LANGUAGE_CODE", "hi-IN")
sarvam_speaker = os.getenv("SARVAM_SPEAKER", "shubh")
gemini_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
gemini_thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low")
gemini_thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "-1")
gemini_fallback_model = os.getenv("GEMINI_FALLBACK_LLM_MODEL", "gemini-2.5-flash")

# AWS Bedrock config (used when LLM_PROVIDER=aws). Amazon Nova models are the most
# broadly accessible default; Claude models additionally require enabling model
# access and a US cross-region inference profile (us.anthropic.claude-...) and are
# the ones that support Bedrock latency-optimized inference.
bedrock_model = os.getenv("AWS_BEDROCK_LLM_MODEL", "amazon.nova-lite-v1:0")
bedrock_region = os.getenv("AWS_BEDROCK_REGION", "us-east-1")

# Groq config (used when LLM_PROVIDER=groq). Groq's very high throughput makes it
# a strong low-latency choice; llama-3.1-8b-instant is the fastest default.
groq_model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
