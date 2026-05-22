import json
from types import SimpleNamespace

import pytest

from tools import dreamlaunch


class FakeRedis:
    def __init__(self) -> None:
        self.calls = []

    def set(self, key, value, ex=None):
        self.calls.append({"key": key, "value": value, "ex": ex})
        return True


def test_save_completed_lead_requires_confirmed_email() -> None:
    with pytest.raises(ValueError, match="Email was not confirmed"):
        dreamlaunch.save_completed_lead_to_redis(
            room_name="woice-call-1",
            caller_number="+918200962735",
            name="Amon",
            email="amon@example.com",
            company="Arisyn",
            reason_for_meet="Wants to build an MVP",
            email_confirmed=False,
            email_sent=True,
        )


def test_save_completed_lead_requires_sent_email() -> None:
    with pytest.raises(ValueError, match="Email was not sent"):
        dreamlaunch.save_completed_lead_to_redis(
            room_name="woice-call-1",
            caller_number="+918200962735",
            name="Amon",
            email="amon@example.com",
            company="Arisyn",
            reason_for_meet="Wants to build an MVP",
            email_confirmed=True,
            email_sent=False,
        )


def test_save_completed_lead_stores_minimal_completed_schema(monkeypatch) -> None:
    fake_redis = FakeRedis()
    monkeypatch.setenv("LEAD_TTL_SECONDS", "86400")
    monkeypatch.setattr(dreamlaunch, "_redis_client", lambda: fake_redis)
    monkeypatch.setattr(
        dreamlaunch.uuid, "uuid4", lambda: SimpleNamespace(hex="abc123def4567890")
    )
    monkeypatch.setattr(dreamlaunch.time, "time", lambda: 1710000000)

    lead = dreamlaunch.save_completed_lead_to_redis(
        room_name="woice-call-1",
        caller_number="+918200962735",
        name="Amon",
        email="amon@example.com",
        company="Arisyn",
        reason_for_meet="Wants to build an MVP",
        email_confirmed=True,
        email_sent=True,
    )

    assert lead == {
        "lead_id": "lead_abc123def456",
        "room_name": "woice-call-1",
        "caller_number": "+918200962735",
        "name": "Amon",
        "email": "amon@example.com",
        "company": "Arisyn",
        "reason_for_meet": "Wants to build an MVP",
        "email_confirmed": True,
        "email_sent": True,
        "status": "completed",
        "created_at": 1710000000,
    }
    assert fake_redis.calls[0]["key"] == "dreamlaunch:lead:lead_abc123def456"
    assert json.loads(fake_redis.calls[0]["value"]) == lead
    assert fake_redis.calls[0]["ex"] == 86400
    assert fake_redis.calls[1] == {
        "key": "dreamlaunch:room:woice-call-1:lead",
        "value": "lead_abc123def456",
        "ex": 86400,
    }


@pytest.mark.asyncio
async def test_send_confirmed_lead_email_and_save_does_not_run_without_permission() -> (
    None
):
    result = await dreamlaunch.send_confirmed_lead_email_and_save._func(
        None,
        name="Amon",
        email="amon@example.com",
        company="Arisyn",
        reason_for_meet="Wants to build an MVP",
        email_confirmed=False,
    )

    assert "has not confirmed permission" in result


@pytest.mark.asyncio
async def test_send_confirmed_lead_email_and_save_sends_then_saves(monkeypatch) -> None:
    events = []

    class FakeParticipant:
        identity = "sip_+918200962735"

    class FakeRoom:
        def __init__(self) -> None:
            self.name = "woice-call-1"
            self.remote_participants = {"caller": FakeParticipant()}

    class FakeSpeech:
        async def wait_for_playout(self):
            events.append("played")

    class FakeSession:
        def __init__(self) -> None:
            self.room = FakeRoom()

        def say(self, text, *, allow_interruptions, add_to_chat_ctx):
            events.append(("say", text, allow_interruptions, add_to_chat_ctx))
            return FakeSpeech()

    class FakeContext:
        def __init__(self) -> None:
            self.session = FakeSession()

    def fake_send_email(lead):
        events.append(("email", lead))

    def fake_save(**kwargs):
        events.append(("redis", kwargs))
        return {"lead_id": "lead_demo"}

    async def fake_end_call(context):
        events.append(("end", context))

    monkeypatch.setattr(dreamlaunch, "send_dreamlaunch_recap_email", fake_send_email)
    monkeypatch.setattr(dreamlaunch, "save_completed_lead_to_redis", fake_save)
    monkeypatch.setattr(dreamlaunch, "_end_room_after_playout", fake_end_call)

    result = await dreamlaunch.send_confirmed_lead_email_and_save._func(
        FakeContext(),
        name="Amon",
        email="amon@example.com",
        company="Arisyn",
        reason_for_meet="Wants to build an MVP",
        email_confirmed=True,
    )

    assert result == "Email sent and lead saved. Lead ID: lead_demo"
    assert events[0] == (
        "email",
        {
            "name": "Amon",
            "email": "amon@example.com",
            "company": "Arisyn",
            "reason_for_meet": "Wants to build an MVP",
        },
    )
    assert events[1][0] == "redis"
    assert events[1][1]["room_name"] == "woice-call-1"
    assert events[1][1]["caller_number"] == "+918200962735"
    assert events[2][0] == "say"
    assert events[3] == "played"
    assert events[4][0] == "end"
