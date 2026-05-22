import json
from types import SimpleNamespace

import pytest

# Imported under its historical name; the Redis namespace is still "dreamlaunch:".
from tools import company as dreamlaunch


class FakeRedis:
    def __init__(self) -> None:
        self.calls = []

    def set(self, key, value, ex=None):
        self.calls.append({"key": key, "value": value, "ex": ex})
        return True


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("amon sharma 2000 at gmail dot com", "amonsharma2000@gmail.com"),
        ("amon@example.com", "amon@example.com"),
        ("Amon @ Gmail . com", "amon@gmail.com"),
        ("john dot doe at the rate company dot co dot uk", "john.doe@company.co.uk"),
        ("amon underscore s at gmail dot com", "amon_s@gmail.com"),
        ("", ""),
    ],
)
def test_normalize_spoken_email(spoken, expected) -> None:
    assert dreamlaunch.normalize_spoken_email(spoken) == expected


@pytest.mark.parametrize(
    "email, valid",
    [
        ("amon@example.com", True),
        ("amonsharma2000@gmail.com", True),
        ("amon at gmail dot com", False),
        ("amon@", False),
        ("amon@gmail", False),
        ("@gmail.com", False),
        ("", False),
    ],
)
def test_is_valid_email(email, valid) -> None:
    assert dreamlaunch.is_valid_email(email) is valid


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+91 82009 62735", "+918200962735"),
        ("plus 91 8200962735", "+918200962735"),
        ("820-096-2735", "8200962735"),
        ("", ""),
    ],
)
def test_normalize_phone(raw, expected) -> None:
    assert dreamlaunch.normalize_phone(raw) == expected


@pytest.mark.parametrize(
    "phone, valid",
    [
        ("+918200962735", True),
        ("8200962", True),
        ("12345", False),
        ("", False),
    ],
)
def test_is_valid_phone(phone, valid) -> None:
    assert dreamlaunch.is_valid_phone(phone) is valid


@pytest.mark.asyncio
async def test_send_confirmed_lead_rejects_malformed_email(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(
        dreamlaunch,
        "send_dreamlaunch_recap_email",
        lambda lead: sent.append(lead),
    )

    result = await dreamlaunch.send_confirmed_lead_email_and_save._func(
        None,
        name="Amon",
        email="amon at gmail",
        company="Arisyn",
        reason_for_meet="Wants to build an MVP",
        email_confirmed=True,
    )

    assert "malformed" in result
    assert sent == []


@pytest.mark.asyncio
async def test_send_confirmed_lead_normalizes_spoken_email(monkeypatch) -> None:
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
            return FakeSpeech()

    class FakeContext:
        def __init__(self) -> None:
            self.session = FakeSession()

    monkeypatch.setattr(
        dreamlaunch,
        "send_dreamlaunch_recap_email",
        lambda lead: events.append(("email", lead["email"])),
    )
    monkeypatch.setattr(
        dreamlaunch, "save_completed_lead_to_redis", lambda **kw: {"lead_id": "lead_x"}
    )

    async def fake_end(context):
        events.append("end")

    monkeypatch.setattr(dreamlaunch, "_end_room_after_playout", fake_end)

    result = await dreamlaunch.send_confirmed_lead_email_and_save._func(
        FakeContext(),
        name="Amon",
        email="amon sharma 2000 at gmail dot com",
        company="Arisyn",
        reason_for_meet="Wants to build an MVP",
        email_confirmed=True,
    )

    assert "Email sent and lead saved" in result
    assert ("email", "amonsharma2000@gmail.com") in events


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


def test_smtp_config_prefers_google_app_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_APP_EMAIL", "hello@dreamlaunch.studio")
    monkeypatch.setenv("GOOGLE_APP_PASS", "app-password")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)

    assert dreamlaunch._smtp_config() == {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "hello@dreamlaunch.studio",
        "password": "app-password",
        "sender": "hello@dreamlaunch.studio",
        "reply_to": "hello@dreamlaunch.studio",
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
