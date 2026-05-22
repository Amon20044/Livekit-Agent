from types import SimpleNamespace

import pytest

from app.dtmf import DtmfCollector, register_dtmf_collector
from tools import get_dialed_phone_number


def test_collector_accumulates_digits() -> None:
    collector = DtmfCollector()
    for digit in "8200962735":
        collector.feed(digit)

    assert collector.digits == "8200962735"
    assert collector.completed is False


def test_pound_marks_complete_and_star_clears() -> None:
    collector = DtmfCollector()
    for digit in "918":
        collector.feed(digit)
    collector.feed("#")
    assert collector.completed is True
    assert collector.digits == "918"

    collector.feed("*")
    assert collector.digits == ""
    assert collector.completed is False


def test_collector_ignores_blank_and_non_digit_noise() -> None:
    collector = DtmfCollector()
    collector.feed("")
    collector.feed(None)  # type: ignore[arg-type]
    collector.feed("9")
    collector.feed("a")
    assert collector.digits == "9"


def test_pound_with_no_digits_does_not_complete() -> None:
    collector = DtmfCollector()
    collector.feed("#")
    assert collector.completed is False


def test_register_forwards_room_events_to_collector() -> None:
    handlers: dict = {}

    class FakeRoom:
        def on(self, event, callback):
            handlers[event] = callback

    collector = DtmfCollector()
    room = FakeRoom()
    register_dtmf_collector(room, collector)

    assert "sip_dtmf_received" in handlers
    handlers["sip_dtmf_received"](SimpleNamespace(digit="7"))
    handlers["sip_dtmf_received"](SimpleNamespace(digit="3"))
    assert collector.digits == "73"


@pytest.mark.asyncio
async def test_get_dialed_phone_number_reports_no_keypad_off_session() -> None:
    context = SimpleNamespace(session=SimpleNamespace(userdata=None))
    result = await get_dialed_phone_number._func(context)
    assert "isn't available" in result


@pytest.mark.asyncio
async def test_get_dialed_phone_number_prompts_when_empty() -> None:
    context = SimpleNamespace(
        session=SimpleNamespace(userdata={"dtmf": DtmfCollector()})
    )
    result = await get_dialed_phone_number._func(context)
    assert "No keypad digits yet" in result


@pytest.mark.asyncio
async def test_get_dialed_phone_number_reads_back_digits() -> None:
    collector = DtmfCollector()
    for digit in "918":
        collector.feed(digit)
    collector.feed("#")
    context = SimpleNamespace(session=SimpleNamespace(userdata={"dtmf": collector}))

    result = await get_dialed_phone_number._func(context)

    assert "918" in result
    assert "9 1 8" in result
    assert "finished entering" in result
