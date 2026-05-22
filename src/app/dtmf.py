"""Collect DTMF keypad input from SIP callers.

Phone callers can't type into a chat box, but they can press digits. LiveKit
emits a ``sip_dtmf_received`` room event for each key. We accumulate those into a
phone number the agent can read back, which is far more reliable than trying to
transcribe a number spoken aloud.

Keypad conventions: ``#`` submits the number, ``*`` clears it and starts over.
"""

import logging

logger = logging.getLogger("agent")

_SUBMIT_KEY = "#"
_CLEAR_KEY = "*"


class DtmfCollector:
    """Accumulates DTMF keypad digits from a SIP caller into a phone number."""

    def __init__(self) -> None:
        self._digits: list[str] = []
        self.completed = False

    def feed(self, digit: str) -> None:
        digit = (digit or "").strip()
        if not digit:
            return
        if digit == _SUBMIT_KEY:
            self.completed = bool(self._digits)
            return
        if digit == _CLEAR_KEY:
            self.clear()
            return
        if digit.isdigit():
            self._digits.append(digit)
            self.completed = False

    @property
    def digits(self) -> str:
        return "".join(self._digits)

    def clear(self) -> None:
        self._digits.clear()
        self.completed = False


def register_dtmf_collector(room, collector: DtmfCollector) -> None:
    """Forward a room's SIP DTMF events into ``collector``."""

    def _on_dtmf(event) -> None:
        collector.feed(getattr(event, "digit", "") or "")

    room.on("sip_dtmf_received", _on_dtmf)
