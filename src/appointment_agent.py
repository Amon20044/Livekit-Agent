from livekit.agents import Agent, RunContext, function_tool
import logging
from datetime import datetime, timedelta
from tools import end_call

logger = logging.getLogger("appointment-agent")

# Fabricated appointment slots in "2025 November 22" format
AVAILABLE_SLOTS = [
    "2025 November 21 10:00 AM",
    "2025 November 21 2:00 PM",
    "2025 November 22 9:00 AM",
    "2025 November 22 3:00 PM",
    "2025 November 23 11:00 AM",
    "2025 November 24 10:30 AM",
    "2025 November 24 4:00 PM",
    "2025 November 25 9:30 AM",
    "2025 November 25 2:30 PM",
    "2025 November 26 11:00 AM",
    "2025 November 26 3:30 PM",
    "2025 November 27 10:00 AM",
    "2025 November 27 4:00 PM",
    "2025 November 28 9:00 AM",
    "2025 November 28 2:00 PM"
]

class AppointmentAgent(Agent):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions="""
You are an appointment booking specialist for PharmaSea Healthcare.

# Introduce
Introduce yourself as Dev Sharma.

# Output rules
- Speak in plain, natural language for voice.
- Keep replies brief (1-2 sentences). Ask one question at a time.
- Be helpful and confirm bookings.

# Tools
- Use available tools to check availability and book appointments.
- Confirm booking details.
- After booking an appointment, always ask if they want to ask something more or end the call.

# Goal
Help patients book appointments and provide confirmation.

# Guardrails
- Only book available slots, suggest alternatives if needed.
""",
            chat_ctx=chat_ctx,
            tools=[end_call]
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="I'm Appointment Agent at PharmaSea Healthcare. I'm Dev Sharma, here to help you book your appointment. What date would you prefer?"
        )

    @function_tool()
    async def get_current_date_time(self, context: RunContext):
        """Get the current date and time in readable format and store in context."""
        now = datetime.now()
        formatted_date = now.strftime("%Y %B %d")  # e.g., "2025 November 22"
        context.userdata.current_date = formatted_date
        context.userdata.current_time = now.strftime("%I:%M %p")  # e.g., "03:45 PM"
        logger.info(f"Current date/time stored: {formatted_date} at {context.userdata.current_time}")
        return f"Current date is {formatted_date}. How can I help with your appointment?"

    @function_tool()
    async def check_availability(self, context: RunContext, date: str):
        """Check available appointment slots for a given date."""
        # For simulation, always show available slots (filter by date or show all)
        available = [slot for slot in AVAILABLE_SLOTS if date in slot]
        if not available:
            # If no exact match, show all slots as available for simulation
            available = AVAILABLE_SLOTS[:5]  # Show first 5 as available
        return f"Available slots: {', '.join(available)}"

    @function_tool()
    async def book_appointment(self, context: RunContext, date_time: str):
        """Book an appointment at the specified date and time."""
        if date_time in AVAILABLE_SLOTS:
            context.userdata.preferred_date = date_time
            logger.info(f"Booked appointment: {date_time}")
            return f"Appointment booked for {date_time}. Confirmation sent to your email. Is there anything else I can help you with today, or would you like to end the call?"
        return f"Sorry, {date_time} is not available. Please choose another time."