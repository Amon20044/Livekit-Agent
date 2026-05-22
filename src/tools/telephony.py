import logging

from livekit.agents import RunContext, function_tool

logger = logging.getLogger(__name__)


def _dtmf_collector(context: RunContext):
    session = getattr(context, "session", None)
    userdata = getattr(session, "userdata", None)
    if isinstance(userdata, dict):
        return userdata.get("dtmf")
    return getattr(userdata, "dtmf", None)


@function_tool
async def get_dialed_phone_number(context: RunContext) -> str:
    """Read back the phone number the caller typed on their phone keypad.

    On a phone call, ask the caller to enter their number on the keypad and press
    the pound (#) key, then call this to retrieve the digits. Always read the
    number back one digit at a time to confirm before using it. Keypad entry is
    not available on web sessions.
    """
    collector = _dtmf_collector(context)
    if collector is None:
        return (
            "Keypad entry isn't available on this session. Ask the caller to type "
            "their number in the chat or say it slowly instead."
        )

    digits = collector.digits
    if not digits:
        return (
            "No keypad digits yet. Ask the caller to enter their phone number on "
            "the keypad and press the pound key when done."
        )

    spaced = " ".join(digits)
    state = "finished entering" if collector.completed else "still entering"
    return (
        f"The caller is {state} their number. Digits so far: {digits}. "
        f"Read this back one digit at a time to confirm: {spaced}."
    )
