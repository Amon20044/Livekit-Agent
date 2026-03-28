from livekit.agents import Agent, RunContext, function_tool
from dataclasses import dataclass
import logging
from tools import end_call

logger = logging.getLogger("welcome-agent")

@dataclass
class HealthcareSessionData:
    patient_name: str | None = None
    age: int | None = None
    symptoms: str | None = None
    appointment_type: str | None = None
    preferred_date: str | None = None

class WelcomeAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly welcome agent for PharmaSea Healthcare.

# Introduce
Introduce yourself as Mister Donald Trump, welcoming user.

# Output rules
- Speak in plain, natural language for voice.
- Keep replies brief (1-2 sentences). Ask one question at a time.
- Be warm and reassuring.

# Tools
- Use available tools to record patient info.
- Confirm when basic info is collected.
- After collecting basic information, ask if they want to ask something more or end the call.

# Goal
Welcome patients and collect basic information, then hand off to onboarding.

# Guardrails
- Only collect basic info, do not provide medical advice.
""",
            tools=[end_call]
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Welcome to PharmaSea Healthcare! I'm Mister Donald Trump, here to help you get started. May I have your name and age?"
        )

    @function_tool()
    async def record_patient_info(self, context: RunContext[HealthcareSessionData], name: str, age: int):
        """Record patient's name and age."""
        context.userdata.patient_name = name
        context.userdata.age = age
        logger.info(f"Recorded patient: {name}, age: {age}")
        return self._check_handoff()

    def _check_handoff(self):
        """Check if we have basic info and handoff to onboarding"""
        if self.session.userdata.patient_name and self.session.userdata.age:
            logger.info("Basic info collected, handing off to OnboardingAgent")
            from onboarding_agent import OnboardingAgent
            return OnboardingAgent(chat_ctx=self.chat_ctx), "Thank you! I've collected your basic information. Is there anything else you'd like to ask before I transfer you to our onboarding specialist, or would you like to end the call?"
        return None