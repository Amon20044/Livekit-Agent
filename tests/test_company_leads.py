import asyncio
import json
from types import SimpleNamespace

import pytest

from tools import company as woice


class FakeKV:
    """In-memory stand-in for the Redis client used by checkpoint tests."""

    def __init__(self, initial: dict | None = None) -> None:
        self.store: dict[str, str] = dict(initial or {})

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        return True


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
        # Digits dictated one at a time ("2 0 0 0") still join with nothing — the
        # normalizer never inserts the dots an LLM might invent ("amon.sharma.2000").
        ("amon sharma 2 0 0 0 at gmail dot com", "amonsharma2000@gmail.com"),
        ("amon@example.com", "amon@example.com"),
        ("Amon @ Gmail . com", "amon@gmail.com"),
        ("john dot doe at the rate company dot co dot uk", "john.doe@company.co.uk"),
        ("amon underscore s at gmail dot com", "amon_s@gmail.com"),
        ("", ""),
    ],
)
def test_normalize_spoken_email(spoken, expected) -> None:
    assert woice.normalize_spoken_email(spoken) == expected


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
    assert woice.is_valid_email(email) is valid


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
    assert woice.normalize_phone(raw) == expected


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
    assert woice.is_valid_phone(phone) is valid


@pytest.mark.asyncio
async def test_send_confirmed_lead_rejects_malformed_email(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(
        woice,
        "send_woice_waitlist_email",
        lambda lead: sent.append(lead),
    )

    result = await woice.send_confirmed_lead_email_and_save._func(
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
        woice,
        "send_woice_waitlist_email",
        lambda lead: events.append(("email", lead["email"])),
    )
    monkeypatch.setattr(
        woice, "save_completed_lead_to_redis", lambda **kw: {"lead_id": "lead_x"}
    )
    monkeypatch.setattr(woice, "save_caller_memory", lambda **kw: None)

    async def fake_end(context):
        events.append("end")

    monkeypatch.setattr(woice, "_end_room_after_playout", fake_end)

    result = await woice.send_confirmed_lead_email_and_save._func(
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
        woice.save_completed_lead_to_redis(
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
        woice.save_completed_lead_to_redis(
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
    monkeypatch.setattr(woice, "_redis_client", lambda: fake_redis)
    monkeypatch.setattr(
        woice.uuid, "uuid4", lambda: SimpleNamespace(hex="abc123def4567890")
    )
    monkeypatch.setattr(woice.time, "time", lambda: 1710000000)

    lead = woice.save_completed_lead_to_redis(
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
    assert fake_redis.calls[0]["key"] == "woice:lead:lead_abc123def456"
    assert json.loads(fake_redis.calls[0]["value"]) == lead
    assert fake_redis.calls[0]["ex"] == 86400
    assert fake_redis.calls[1] == {
        "key": "woice:room:woice-call-1:lead",
        "value": "lead_abc123def456",
        "ex": 86400,
    }


def test_smtp_config_prefers_google_app_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_APP_EMAIL", "hello@woice.ai")
    monkeypatch.setenv("GOOGLE_APP_PASS", "app-password")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)

    assert woice._smtp_config() == {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "hello@woice.ai",
        "password": "app-password",
        "sender": "hello@woice.ai",
        "reply_to": "hello@woice.ai",
    }


def test_waitlist_email_html_is_branded_and_escapes_user_text(monkeypatch) -> None:
    monkeypatch.setenv("COMPANY_NAME", "Woice AI")
    monkeypatch.setenv("COMPANY_WEBSITE", "https://woice.vercel.app")

    html = woice._email_html_body(
        {
            "name": "Amon <script>",
            "email": "amon@example.com",
            "company": "Arisyn",
            "reason_for_meet": "Book calls <fast>",
        }
    )

    assert "Welcome to Woice AI" in html
    assert "You are on the Woice AI waitlist" in html
    assert "Voice that does not miss a thing" in html
    assert "https://woice.vercel.app" in html
    assert "Amon &lt;script&gt;" in html
    assert "Book calls &lt;fast&gt;" in html
    assert "Amon <script>" not in html


def test_send_woice_waitlist_email_builds_multipart_message(monkeypatch) -> None:
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def starttls(self, context):
            return None

        def login(self, username, password):
            return None

        def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr(
        woice,
        "_smtp_config",
        lambda: {
            "host": "smtp.example.com",
            "port": 587,
            "username": "hello@woice.ai",
            "password": "secret",
            "sender": "hello@woice.ai",
            "reply_to": "founders@woice.ai",
        },
    )
    monkeypatch.setattr(woice.smtplib, "SMTP", FakeSMTP)

    woice.send_woice_waitlist_email(
        {
            "name": "Amon",
            "email": "amon@example.com",
            "company": "Arisyn",
            "reason_for_meet": "Qualify inbound leads and book demos",
        }
    )

    assert len(sent) == 1
    message = sent[0]
    assert message["Subject"] == "Your Woice AI waitlist recap"
    assert message["Reply-To"] == "founders@woice.ai"
    assert message.is_multipart()
    assert message.get_body(preferencelist=("plain",)).get_content_type() == (
        "text/plain"
    )
    html = message.get_body(preferencelist=("html",)).get_content()
    assert "Qualify inbound leads and book demos" in html
    assert "What happens next" in html


@pytest.mark.asyncio
async def test_send_confirmed_lead_email_and_save_does_not_run_without_permission() -> (
    None
):
    result = await woice.send_confirmed_lead_email_and_save._func(
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

    caller_saves = []
    monkeypatch.setattr(woice, "send_woice_waitlist_email", fake_send_email)
    monkeypatch.setattr(woice, "save_completed_lead_to_redis", fake_save)
    monkeypatch.setattr(woice, "_end_room_after_playout", fake_end_call)
    monkeypatch.setattr(
        woice, "save_caller_memory", lambda **kwargs: caller_saves.append(kwargs)
    )

    result = await woice.send_confirmed_lead_email_and_save._func(
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
    assert caller_saves == [
        {
            "phone": "+918200962735",
            "status": "completed",
            "name": "Amon",
            "email": "amon@example.com",
            "company": "Arisyn",
            "reason_for_meet": "Wants to build an MVP",
            "lead_id": "lead_demo",
        }
    ]


def test_caller_number_from_room_reads_sip_identity() -> None:
    class FakeParticipant:
        identity = "sip_+918200962735"

    class FakeRoom:
        def __init__(self) -> None:
            self.remote_participants = {"caller": FakeParticipant()}

    assert woice.caller_number_from_room(FakeRoom()) == "+918200962735"


def test_caller_number_from_room_handles_no_participants() -> None:
    class FakeRoom:
        def __init__(self) -> None:
            self.remote_participants = {}

    assert woice.caller_number_from_room(FakeRoom()) is None


def _room_with(**participant_attrs):
    participant = SimpleNamespace(
        identity=participant_attrs.pop("identity", ""),
        attributes=participant_attrs.pop("attributes", {}),
        metadata=participant_attrs.pop("metadata", None),
    )
    return SimpleNamespace(remote_participants={"p": participant})


def test_caller_number_from_room_reads_sip_phone_attribute() -> None:
    # Dispatch rules can set a custom identity; the number still rides sip.phoneNumber.
    room = _room_with(
        identity="agent-caller-1", attributes={"sip.phoneNumber": "+14155550123"}
    )
    assert woice.caller_number_from_room(room) == "+14155550123"


def test_web_visitor_ip_from_attribute_and_metadata() -> None:
    assert (
        woice.web_visitor_ip_from_room(
            _room_with(attributes={"visitor_ip": "203.0.113.7"})
        )
        == "203.0.113.7"
    )
    # First hop of an X-Forwarded-For style value wins.
    assert (
        woice.web_visitor_ip_from_room(
            _room_with(attributes={"client_ip": "198.51.100.9, 10.0.0.1"})
        )
        == "198.51.100.9"
    )
    assert (
        woice.web_visitor_ip_from_room(
            _room_with(metadata=json.dumps({"ip": "192.0.2.5"}))
        )
        == "192.0.2.5"
    )


def test_web_visitor_ip_rejects_garbage() -> None:
    assert (
        woice.web_visitor_ip_from_room(
            _room_with(attributes={"visitor_ip": "not-an-ip"})
        )
        is None
    )
    assert woice.web_visitor_ip_from_room(_room_with()) is None


def test_caller_ref_prefers_phone_then_ip_then_none() -> None:
    assert (
        woice.caller_ref_from_room(_room_with(identity="sip_+918200962735"))
        == "+918200962735"
    )
    assert (
        woice.caller_ref_from_room(_room_with(attributes={"visitor_ip": "203.0.113.7"}))
        == "ip:203.0.113.7"
    )
    assert woice.caller_ref_from_room(_room_with()) is None


def test_redis_health_check_reports_ok_and_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        woice, "_redis_client", lambda: SimpleNamespace(ping=lambda: True)
    )
    assert woice.redis_health_check() == (True, "ok")

    def boom():
        raise ConnectionError("refused")

    monkeypatch.setattr(woice, "_redis_client", lambda: SimpleNamespace(ping=boom))
    ok, detail = woice.redis_health_check()
    assert ok is False
    assert "refused" in detail


def test_lookup_caller_returns_none_without_phone() -> None:
    assert woice.lookup_caller(None) is None
    assert woice.lookup_caller("") is None


def test_save_and_lookup_caller_roundtrip(monkeypatch) -> None:
    store = {}

    class FakeKV:
        def set(self, key, value, ex=None):
            store[key] = value

        def get(self, key):
            return store.get(key)

    monkeypatch.setattr(woice, "_redis_client", lambda: FakeKV())

    record = woice.save_caller_memory(
        phone="+918200962735", status="partial", name="Amon"
    )
    assert record["status"] == "partial"
    assert record["name"] == "Amon"

    got = woice.lookup_caller("+918200962735")
    assert got["phone"] == "+918200962735"
    assert got["name"] == "Amon"


def test_save_caller_memory_skips_without_phone() -> None:
    assert woice.save_caller_memory(phone=None, status="partial") is None


def test_lookup_caller_survives_redis_failure(monkeypatch) -> None:
    def boom():
        raise RuntimeError("redis down")

    monkeypatch.setattr(woice, "_redis_client", boom)
    assert woice.lookup_caller("+918200962735") is None


@pytest.mark.asyncio
async def test_note_lead_progress_accumulates_in_userdata() -> None:
    userdata = {"lead_progress": {}}
    context = SimpleNamespace(session=SimpleNamespace(userdata=userdata))

    await woice.note_lead_progress._func(context, name="Amon")
    await woice.note_lead_progress._func(
        context, email="amon at gmail dot com", company="Arisyn"
    )

    assert userdata["lead_progress"] == {
        "name": "Amon",
        "email": "amon@gmail.com",
        "company": "Arisyn",
    }


def test_upsert_caller_checkpoint_merges_into_existing_record(monkeypatch) -> None:
    kv = FakeKV()
    monkeypatch.setattr(woice, "_redis_client", lambda: kv)
    monkeypatch.setattr(woice.time, "time", lambda: 1710000000)

    first = woice.upsert_caller_checkpoint(phone="+918200962735", name="Amon")
    assert first["name"] == "Amon"
    assert first["status"] == "partial"
    assert first["email"] is None

    # A later checkpoint that only knows the email must NOT erase the name.
    second = woice.upsert_caller_checkpoint(
        phone="+918200962735", email="amon@example.com", company="Arisyn"
    )
    assert second["name"] == "Amon"
    assert second["email"] == "amon@example.com"
    assert second["company"] == "Arisyn"
    assert second["status"] == "partial"
    assert second["updated_at"] == 1710000000
    assert json.loads(kv.store["woice:caller:+918200962735"]) == second


def test_upsert_caller_checkpoint_does_not_downgrade_completed(monkeypatch) -> None:
    kv = FakeKV(
        {
            "woice:caller:+918200962735": json.dumps(
                {"phone": "+918200962735", "status": "completed", "name": "Amon"}
            )
        }
    )
    monkeypatch.setattr(woice, "_redis_client", lambda: kv)

    record = woice.upsert_caller_checkpoint(
        phone="+918200962735", status="partial", reason_for_meet="More info"
    )

    assert record["status"] == "completed"
    assert record["name"] == "Amon"
    assert record["reason_for_meet"] == "More info"


def test_upsert_caller_checkpoint_skips_without_phone() -> None:
    assert woice.upsert_caller_checkpoint(phone=None, name="Amon") is None
    assert woice.upsert_caller_checkpoint(phone="", name="Amon") is None


def test_upsert_caller_checkpoint_survives_redis_failure(monkeypatch) -> None:
    def boom():
        raise RuntimeError("redis down")

    monkeypatch.setattr(woice, "_redis_client", boom)
    assert woice.upsert_caller_checkpoint(phone="+918200962735", name="Amon") is None


def test_upsert_caller_checkpoint_survives_corrupt_existing_record(monkeypatch) -> None:
    kv = FakeKV({"woice:caller:+918200962735": "{not valid json"})
    monkeypatch.setattr(woice, "_redis_client", lambda: kv)

    record = woice.upsert_caller_checkpoint(phone="+918200962735", name="Amon")

    assert record["name"] == "Amon"
    assert record["status"] == "partial"


@pytest.mark.asyncio
async def test_note_lead_progress_checkpoints_per_number_in_background(
    monkeypatch,
) -> None:
    recorded = []

    def fake_upsert(**kwargs):
        recorded.append(kwargs)
        return kwargs

    monkeypatch.setattr(woice, "upsert_caller_checkpoint", fake_upsert)

    userdata = {"caller_ref": "+918200962735", "lead_progress": {}}
    context = SimpleNamespace(session=SimpleNamespace(userdata=userdata))

    await woice.note_lead_progress._func(
        context, name="Amon", email="amon at gmail dot com"
    )

    # The checkpoint runs in a background task; await it to let it finish.
    tasks = userdata.get("checkpoint_tasks")
    assert tasks
    await asyncio.gather(*list(tasks))

    assert recorded == [
        {
            "phone": "+918200962735",
            "status": "partial",
            "name": "Amon",
            "email": "amon@gmail.com",
        }
    ]


@pytest.mark.asyncio
async def test_note_lead_progress_skips_checkpoint_without_phone(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(
        woice, "upsert_caller_checkpoint", lambda **kw: recorded.append(kw)
    )

    userdata = {"lead_progress": {}}
    context = SimpleNamespace(session=SimpleNamespace(userdata=userdata))

    await woice.note_lead_progress._func(context, name="Amon")

    assert userdata.get("checkpoint_tasks") in (None, set())
    assert recorded == []


@pytest.mark.asyncio
async def test_note_lead_progress_skips_checkpoint_when_completed(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(
        woice, "upsert_caller_checkpoint", lambda **kw: recorded.append(kw)
    )

    userdata = {
        "caller_ref": "+918200962735",
        "lead_progress": {},
        "lead_completed": True,
    }
    context = SimpleNamespace(session=SimpleNamespace(userdata=userdata))

    await woice.note_lead_progress._func(context, name="Amon")

    assert userdata.get("checkpoint_tasks") in (None, set())
    assert recorded == []
