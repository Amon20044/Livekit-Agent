from livekit.agents import Agent, RunContext, function_tool
import logging
from tools import end_call

logger = logging.getLogger("onboarding-agent")

class OnboardingAgent(Agent):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions="""
You are an onboarding specialist for PharmaSea Healthcare.

# Introduce
Introduce yourself as Amon Sharma, an Onboarding specialist

# Output rules
- Speak in plain, natural language for voice.
- Keep replies brief (1-2 sentences). Ask one question at a time.
- Be empathetic and professional.

# Tools
- Use available tools to record symptoms and appointment preferences.
- Confirm when details are collected.
- After collecting all information, ask if they want to ask something more or end the call.

# Goal
Collect symptoms and appointment preferences, then hand off to appointment booking.

# Guardrails
- Do not diagnose, only collect information.
""",
            chat_ctx=chat_ctx,
            tools=[end_call]
        )

    async def on_enter(self) -> None:
        userdata = self.session.userdata
        greeting = f"Hello {userdata.patient_name}!, I am Amon, Onboarding Specialist at PharmaSea Healthcare."
        await self.session.generate_reply(
            instructions=f"{greeting} I'm here to help schedule your appointment. What symptoms are you experiencing?"
        )

    @function_tool()
    async def record_symptoms(self, context: RunContext, symptoms: str):
        """Record patient's symptoms."""
        context.userdata.symptoms = symptoms
        logger.info(f"Recorded symptoms: {symptoms}")
        return self._check_handoff()

    @function_tool()
    async def record_appointment_type(self, context: RunContext, appointment_type: str):
        """Record preferred appointment type (e.g., consultation, checkup)."""
        context.userdata.appointment_type = appointment_type
        logger.info(f"Recorded appointment type: {appointment_type}")
        return self._check_handoff()

    def _check_handoff(self):
        """Check if we have symptoms and type, then handoff to appointment booking"""
        if self.session.userdata.symptoms and self.session.userdata.appointment_type:
            logger.info("Appointment details collected, handing off to AppointmentAgent")
            from appointment_agent import AppointmentAgent
            return AppointmentAgent(chat_ctx=self.chat_ctx), "Thank you for providing your symptoms and appointment preferences. Is there anything else you'd like to ask before I transfer you to appointment booking, or would you like to end the call?"
        return None