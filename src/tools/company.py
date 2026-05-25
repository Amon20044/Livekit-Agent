import asyncio
import ipaddress
import json
import logging
import os
import re
import smtplib
import ssl
import time
import uuid
from email.message import EmailMessage
from html import escape
from typing import Any

from livekit.agents import RunContext, function_tool, get_job_context
from redis import Redis

logger = logging.getLogger(__name__)

# Basic, deliberately permissive email shape check. We are guarding against
# obviously-broken transcripts ("amon at gmail dot com" left unconverted, missing
# domains, stray spaces), not enforcing the full RFC.
_EMAIL_RE = re.compile(r"^[a-z0-9!#$%&'*+/=?^_`{|}~.-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+$")

# Spoken-symbol words, longest first so "at the rate" wins over "at".
_SPOKEN_EMAIL_SUBSTITUTIONS = (
    (r"\b(?:at the rate(?: of)?|ऐट द रेट|एट द रेट|एट द रेट ऑफ)\b", "@"),
    (r"\b(?:underscore|अंडरस्कोर)\b", "_"),
    (r"\b(?:dash|hyphen|minus|डैश|हाइफ़न|हाइफन|माइनस)\b", "-"),
    (r"\b(?:plus|प्लस)\b", "+"),
    (r"\b(?:dot|डॉट)\b", "."),
    (r"\b(?:at|ऐट|एट)\b", "@"),
)

_SPOKEN_EMAIL_DIGITS = (
    (r"\b(?:two thousand|two thousands|दो हजार)\b", "2000"),
    (r"\b(?:zero|जीरो|ज़ीरो|शून्य)\b", "0"),
    (r"\b(?:one|वन|एक)\b", "1"),
    (r"\b(?:two|टू|दो)\b", "2"),
    (r"\b(?:three|थ्री|तीन)\b", "3"),
    (r"\b(?:four|फोर|चार)\b", "4"),
    (r"\b(?:five|फाइव|पांच|पाँच)\b", "5"),
    (r"\b(?:six|सिक्स|छह|छः)\b", "6"),
    (r"\b(?:seven|सेवन|सात)\b", "7"),
    (r"\b(?:eight|एट|आठ)\b", "8"),
    (r"\b(?:nine|नाइन|नौ)\b", "9"),
)


def normalize_spoken_email(raw: str) -> str:
    """Turn a spoken/typed email into a canonical address.

    Converts dictated symbols ("amon at gmail dot com" -> "amon@gmail.com"),
    drops the spaces speech-to-text sprinkles in, and lowercases. Returns the
    best-effort result; callers should validate with `is_valid_email`.
    """
    if not raw:
        return ""

    text = raw.strip().lower()
    for pattern, replacement in _SPOKEN_EMAIL_SUBSTITUTIONS:
        text = re.sub(pattern, f" {replacement} ", text)
    for pattern, replacement in _SPOKEN_EMAIL_DIGITS:
        text = re.sub(pattern, f" {replacement} ", text)

    text = re.sub(r"\s+", "", text)
    # Collapse separators that spacing/substitution may have duplicated.
    text = re.sub(r"@{2,}", "@", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip(".")


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


def format_email_for_readback(email: str) -> str:
    """Space an email out so the agent can read it back unambiguously."""
    return " ".join((email or "").strip())


def normalize_phone(raw: str) -> str:
    """Reduce a dictated/typed phone number to digits, preserving a leading +."""
    if not raw:
        return ""

    text = raw.strip().lower().replace("plus", "+")
    keep_plus = text.lstrip().startswith("+")
    digits = re.sub(r"\D", "", text)
    return f"+{digits}" if keep_plus and digits else digits


def is_valid_phone(phone: str) -> bool:
    digits = re.sub(r"\D", "", phone or "")
    # E.164 allows up to 15 digits; 7 is a safe lower bound for a real number.
    return 7 <= len(digits) <= 15


def _clean_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


def _lead_ttl_seconds() -> int:
    raw_value = _clean_env("LEAD_TTL_SECONDS")
    if raw_value is None:
        return 86_400

    try:
        return max(60, int(raw_value))
    except ValueError:
        logger.warning("Invalid LEAD_TTL_SECONDS=%r; using 86400", raw_value)
        return 86_400


def _redis_url() -> str:
    return _clean_env("REDIS_URL") or "redis://localhost:6379/0"


def _redis_client() -> Redis:
    return Redis.from_url(_redis_url(), decode_responses=True)


def redis_health_check() -> tuple[bool, str]:
    """Ping Redis so callers can surface an unreachable store at startup.

    Caller memory and per-call checkpoints swallow every Redis error so a storage
    outage never breaks a live call — which also means resume silently does nothing
    when Redis is down. This check lets the worker log one clear warning instead.
    Returns ``(ok, detail)`` and never raises.
    """
    try:
        _redis_client().ping()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def save_completed_lead_to_redis(
    *,
    room_name: str,
    caller_number: str | None,
    name: str,
    email: str,
    company: str | None,
    reason_for_meet: str,
    email_confirmed: bool,
    email_sent: bool,
) -> dict[str, Any]:
    if not email_confirmed:
        raise ValueError("Email was not confirmed by caller")

    if not email_sent:
        raise ValueError("Email was not sent yet")

    lead_id = f"lead_{uuid.uuid4().hex[:12]}"
    now = int(time.time())
    lead = {
        "lead_id": lead_id,
        "room_name": room_name,
        "caller_number": caller_number,
        "name": name,
        "email": email,
        "company": company,
        "reason_for_meet": reason_for_meet,
        "email_confirmed": True,
        "email_sent": True,
        "status": "completed",
        "created_at": now,
    }

    ttl_seconds = _lead_ttl_seconds()
    redis = _redis_client()
    redis.set(
        f"woice:lead:{lead_id}",
        json.dumps(lead),
        ex=ttl_seconds,
    )
    redis.set(
        f"woice:room:{room_name}:lead",
        lead_id,
        ex=ttl_seconds,
    )

    return lead


def _caller_memory_ttl_seconds() -> int:
    raw_value = _clean_env("CALLER_MEMORY_TTL_SECONDS")
    if raw_value is None:
        return 2_592_000  # 30 days

    try:
        return max(3_600, int(raw_value))
    except ValueError:
        logger.warning("Invalid CALLER_MEMORY_TTL_SECONDS=%r; using 2592000", raw_value)
        return 2_592_000


def _caller_key(phone: str) -> str:
    return f"woice:caller:{phone}"


def lookup_caller(phone: str | None) -> dict[str, Any] | None:
    """Return a stored record for a returning caller, keyed by phone, or None.

    Never raises: a Redis hiccup must not stop a live call from connecting.
    """
    if not phone:
        return None
    try:
        raw = _redis_client().get(_caller_key(phone))
    except Exception:
        logger.warning("Caller memory lookup failed for %s", phone, exc_info=True)
        return None
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Corrupt caller memory for %s; ignoring", phone)
        return None
    return record if isinstance(record, dict) else None


def save_caller_memory(
    *,
    phone: str | None,
    status: str,
    name: str | None = None,
    email: str | None = None,
    company: str | None = None,
    reason_for_meet: str | None = None,
    lead_id: str | None = None,
) -> dict[str, Any] | None:
    """Persist a phone-indexed caller record so a returning caller is recognized.

    Stores nothing when there is no phone number (e.g. web sessions). Never
    raises on Redis failure; remembering a caller must not break the call.
    """
    if not phone:
        return None

    record = {
        "phone": phone,
        "status": status,
        "name": name,
        "email": email,
        "company": company,
        "reason_for_meet": reason_for_meet,
        "lead_id": lead_id,
        "updated_at": int(time.time()),
    }
    try:
        _redis_client().set(
            _caller_key(phone),
            json.dumps(record),
            ex=_caller_memory_ttl_seconds(),
        )
    except Exception:
        logger.warning("Failed to save caller memory for %s", phone, exc_info=True)
        return None
    return record


_CHECKPOINT_FIELDS = ("name", "email", "company", "reason_for_meet", "lead_id")


def upsert_caller_checkpoint(
    *,
    phone: str | None,
    status: str = "partial",
    name: str | None = None,
    email: str | None = None,
    company: str | None = None,
    reason_for_meet: str | None = None,
    lead_id: str | None = None,
) -> dict[str, Any] | None:
    """Merge new details into the per-number Redis record (an upsert checkpoint).

    Unlike ``save_caller_memory`` (which overwrites the whole record), this reads
    the existing record and merges in only the non-empty fields, so a checkpoint
    that learns just an email never erases a name captured earlier. It is the
    durability backbone for live calls: every time the agent learns something, the
    latest snapshot is persisted, so a dropped call, a rate-limited LLM, or a crash
    still leaves a resumable record keyed by phone number.

    Never raises (Redis must not break a live call) and never downgrades a
    "completed" lead back to "partial". Returns nothing without a phone number
    (e.g. web sessions).
    """
    if not phone:
        return None

    try:
        redis = _redis_client()
    except Exception:
        logger.warning("Checkpoint client init failed for %s", phone, exc_info=True)
        return None

    existing: dict[str, Any] = {}
    try:
        raw = redis.get(_caller_key(phone))
        if raw:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                existing = loaded
    except Exception:
        # A read failure or corrupt record must not stop us writing a fresh one.
        logger.warning(
            "Checkpoint read failed for %s; writing fresh record", phone, exc_info=True
        )

    incoming = {
        "name": name,
        "email": email,
        "company": company,
        "reason_for_meet": reason_for_meet,
        "lead_id": lead_id,
    }

    record = dict(existing)
    record["phone"] = phone
    # Never let a partial checkpoint clobber a finished lead.
    record["status"] = "completed" if existing.get("status") == "completed" else status
    for key in _CHECKPOINT_FIELDS:
        value = incoming.get(key)
        if value:
            record[key] = value
        else:
            record.setdefault(key, existing.get(key))
    record["updated_at"] = int(time.time())

    try:
        redis.set(
            _caller_key(phone),
            json.dumps(record),
            ex=_caller_memory_ttl_seconds(),
        )
    except Exception:
        logger.warning("Checkpoint write failed for %s", phone, exc_info=True)
        return None
    return record


def _required_env(name: str) -> str:
    value = _clean_env(name)
    if not value:
        raise RuntimeError(f"{name} is required to send Woice AI waitlist emails")
    return value


def _smtp_config() -> dict[str, str | int]:
    google_app_email = _clean_env("GOOGLE_APP_EMAIL")
    google_app_pass = _clean_env("GOOGLE_APP_PASS")

    smtp_host = _clean_env("SMTP_HOST") or "smtp.gmail.com"
    smtp_port = int(_clean_env("SMTP_PORT") or "587")
    smtp_username = google_app_email or _required_env("SMTP_USERNAME")
    smtp_password = google_app_pass or _required_env("SMTP_PASSWORD")
    sender = _clean_env("SMTP_FROM") or smtp_username
    reply_to = _clean_env("WOICE_REPLY_TO") or sender

    return {
        "host": smtp_host,
        "port": smtp_port,
        "username": smtp_username,
        "password": smtp_password,
        "sender": sender,
        "reply_to": reply_to,
    }


def company_name() -> str:
    return _clean_env("COMPANY_NAME") or "Woice AI"


def company_website() -> str:
    return _clean_env("COMPANY_WEBSITE") or "https://woice.vercel.app"


def _email_body(lead: dict[str, str | None]) -> str:
    company = company_name()
    website = company_website()
    waitlist_url = _clean_env("WOICE_WAITLIST_URL") or website
    company_line = f"Company: {lead.get('company') or 'Not provided'}"
    return f"""Hi {lead["name"]},

Thanks for joining the {company} waitlist. Here is the voice workflow brief I captured:

Name: {lead["name"]}
Email: {lead["email"]}
{company_line}
Workflow to automate: {lead["reason_for_meet"]}

Next step:
We will review your use case and follow up with the right onboarding path.
Waitlist page: {waitlist_url}

{company} connects phone calls to business tools so every conversation can qualify, book, update, remind, escalate, and close the loop automatically.

{company}
{website}
"""


def _email_html_body(lead: dict[str, str | None]) -> str:
    company = escape(company_name())
    website = escape(company_website())
    waitlist_url = escape(_clean_env("WOICE_WAITLIST_URL") or company_website())
    name = escape(str(lead["name"]))
    email = escape(str(lead["email"]))
    caller_company = escape(str(lead.get("company") or "Not provided"))
    workflow = escape(str(lead["reason_for_meet"]))

    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{company} waitlist recap</title>
  </head>
  <body style="margin:0; padding:0; background:#07090d; color:#f6f7fb; font-family:Inter, Arial, sans-serif;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#07090d; padding:32px 12px;">
      <tr>
        <td align="center">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:640px; border:1px solid #202633; border-radius:24px; overflow:hidden; background:#0d1118;">
            <tr>
              <td style="padding:34px 30px 26px; background:linear-gradient(135deg,#101826 0%,#0d1118 48%,#11251f 100%);">
                <div style="font-size:13px; letter-spacing:0.14em; text-transform:uppercase; color:#8be7c2; font-weight:700;">Welcome to {company}</div>
                <h1 style="margin:18px 0 10px; font-size:34px; line-height:1.08; color:#ffffff; font-weight:800;">You are on the {company} waitlist.</h1>
                <p style="margin:0; max-width:520px; color:#c9d2df; font-size:16px; line-height:1.65;">Voice that does not miss a thing, and actually finishes the conversation. We captured your workflow brief and will use it to shape the right onboarding path for you.</p>
              </td>
            </tr>
            <tr>
              <td style="padding:26px 30px 6px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="padding:16px 0; border-bottom:1px solid #202633;">
                      <div style="color:#8a96a8; font-size:12px; text-transform:uppercase; letter-spacing:0.12em;">Name</div>
                      <div style="margin-top:6px; color:#ffffff; font-size:17px; font-weight:700;">{name}</div>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:16px 0; border-bottom:1px solid #202633;">
                      <div style="color:#8a96a8; font-size:12px; text-transform:uppercase; letter-spacing:0.12em;">Email</div>
                      <div style="margin-top:6px; color:#ffffff; font-size:17px; font-weight:700;">{email}</div>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:16px 0; border-bottom:1px solid #202633;">
                      <div style="color:#8a96a8; font-size:12px; text-transform:uppercase; letter-spacing:0.12em;">Company</div>
                      <div style="margin-top:6px; color:#ffffff; font-size:17px; font-weight:700;">{caller_company}</div>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:16px 0;">
                      <div style="color:#8a96a8; font-size:12px; text-transform:uppercase; letter-spacing:0.12em;">Workflow to automate</div>
                      <div style="margin-top:8px; color:#eaf0f7; font-size:16px; line-height:1.65;">{workflow}</div>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr>
              <td style="padding:18px 30px 30px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#111821; border:1px solid #263244; border-radius:18px;">
                  <tr>
                    <td style="padding:22px;">
                      <div style="color:#8be7c2; font-size:14px; font-weight:800;">What happens next</div>
                      <p style="margin:10px 0 0; color:#c9d2df; font-size:15px; line-height:1.65;">We will review your use case for call volume, integrations, language needs, and the outcome your agent should complete. When onboarding opens, you will hear from us with the best setup path.</p>
                    </td>
                  </tr>
                </table>
                <div style="padding-top:24px;">
                  <a href="{waitlist_url}" style="display:inline-block; background:#8be7c2; color:#07100d; text-decoration:none; font-weight:800; font-size:15px; padding:14px 20px; border-radius:999px;">Visit Woice</a>
                  <span style="display:inline-block; margin-left:12px; color:#8a96a8; font-size:14px;">{website}</span>
                </div>
              </td>
            </tr>
            <tr>
              <td style="padding:22px 30px; background:#090c12; border-top:1px solid #202633;">
                <p style="margin:0; color:#8a96a8; font-size:13px; line-height:1.7;">{company} turns phone calls into completed workflows across CRM, calendar, knowledge base, payments, support, messaging, and internal tools.</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""


def send_woice_waitlist_email(lead: dict[str, str | None]) -> None:
    smtp_config = _smtp_config()

    message = EmailMessage()
    message["Subject"] = f"Your {company_name()} waitlist recap"
    message["From"] = str(smtp_config["sender"])
    message["To"] = str(lead["email"])
    message["Reply-To"] = str(smtp_config["reply_to"])
    message.set_content(_email_body(lead))
    message.add_alternative(_email_html_body(lead), subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP(
        str(smtp_config["host"]),
        int(smtp_config["port"]),
        timeout=15,
    ) as smtp:
        smtp.starttls(context=context)
        smtp.login(str(smtp_config["username"]), str(smtp_config["password"]))
        smtp.send_message(message)


def _get_room_name(context: RunContext) -> str:
    room = getattr(context.session, "room", None)
    return getattr(room, "name", None) or "unknown-room"


def caller_number_from_room(room) -> str | None:
    """Extract the caller's phone number from a room's SIP participant.

    Prefers the dispatch-rule identity (``sip_<number>`` or a raw ``+<number>``)
    and falls back to the ``sip.phoneNumber`` / ``sip.from`` participant attribute
    that LiveKit SIP sets even when the identity is customized.
    """
    participants = getattr(room, "remote_participants", {}) or {}

    for participant in participants.values():
        identity = getattr(participant, "identity", "") or ""
        if identity.startswith("sip_"):
            return identity.removeprefix("sip_")
        if identity.startswith("+"):
            return identity

        attributes = getattr(participant, "attributes", None) or {}
        number = attributes.get("sip.phoneNumber") or attributes.get("sip.from")
        if number:
            return number.strip()

    return None


# Participant attribute / metadata keys a web frontend can use to hand the
# visitor's IP to the agent (WebRTC does not expose the client IP otherwise).
_WEB_IP_KEYS = ("visitor_ip", "client_ip", "remote_ip", "ip")


def _normalize_ip(value: str | None) -> str | None:
    """Return a canonical IP string, taking the first hop of an XFF list."""
    if not value:
        return None
    candidate = value.split(",")[0].strip()
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def web_visitor_ip_from_room(room) -> str | None:
    """Best-effort visitor IP for a web session.

    The browser's IP is not visible to the agent over WebRTC, so the frontend
    must attach it as a participant attribute (or a JSON metadata field) using one
    of ``_WEB_IP_KEYS``. Returns the first valid IP found, or None.
    """
    participants = getattr(room, "remote_participants", {}) or {}

    for participant in participants.values():
        attributes = getattr(participant, "attributes", None) or {}
        for key in _WEB_IP_KEYS:
            ip = _normalize_ip(attributes.get(key))
            if ip:
                return ip

        metadata = getattr(participant, "metadata", None)
        if metadata:
            try:
                meta = json.loads(metadata)
            except (TypeError, ValueError):
                meta = None
            if isinstance(meta, dict):
                for key in _WEB_IP_KEYS:
                    ip = _normalize_ip(meta.get(key))
                    if ip:
                        return ip

    return None


def caller_ref_from_room(room) -> str | None:
    """Stable per-caller key: the phone number for telephony, ``ip:<addr>`` for
    web visitors, or None when neither is available (e.g. console sessions).

    This is what caller memory and checkpoints are keyed on, so a returning phone
    caller or a returning web visitor (same IP) is recognized and can resume.
    """
    phone = caller_number_from_room(room)
    if phone:
        return phone

    ip = web_visitor_ip_from_room(room)
    if ip:
        return f"ip:{ip}"

    return None


def _get_caller_number_from_room(context: RunContext) -> str | None:
    return caller_number_from_room(getattr(context.session, "room", None))


def _get_caller_ref_from_room(context: RunContext) -> str | None:
    return caller_ref_from_room(getattr(context.session, "room", None))


async def _end_room_after_playout(context: RunContext) -> None:
    job_ctx = get_job_context()
    if job_ctx is not None:
        await job_ctx.delete_room()
        return

    shutdown = getattr(context.session, "shutdown", None)
    if callable(shutdown):
        shutdown(drain=True)


def _session_userdata(context: RunContext) -> dict[str, Any] | None:
    userdata = getattr(getattr(context, "session", None), "userdata", None)
    return userdata if isinstance(userdata, dict) else None


def _schedule_caller_checkpoint(userdata: dict[str, Any] | None):
    """Fire-and-forget upsert of the per-number checkpoint to Redis.

    Runs the (blocking) Redis write in a worker thread and does not await it, so
    the live call is never stalled by storage. Because it runs on every detail the
    agent captures, the most recent snapshot is already persisted before any later
    failure — a dropped call, a rate-limited LLM, or a crash leaves a resumable
    record. Returns the scheduled task (or None when there is nothing to persist),
    which the shutdown hook drains.
    """
    if not isinstance(userdata, dict):
        return None

    caller_ref = userdata.get("caller_ref")
    progress = userdata.get("lead_progress") or {}
    if not caller_ref or not progress or userdata.get("lead_completed"):
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    task = loop.create_task(
        asyncio.to_thread(
            upsert_caller_checkpoint, phone=caller_ref, status="partial", **progress
        )
    )
    tasks = userdata.setdefault("checkpoint_tasks", set())
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


@function_tool
async def note_lead_progress(
    context: RunContext,
    name: str | None = None,
    email: str | None = None,
    company: str | None = None,
    reason_for_meet: str | None = None,
) -> str:
    """Remember details captured so far this call and checkpoint them in the background.

    Call this as you learn the caller's name, email, company, or reason for
    meeting. It updates in-memory notes and, on phone calls, upserts a per-number
    checkpoint to storage in the background so a returning caller can resume even
    if this call drops before completion. It sends no email and never blocks the
    conversation.
    """
    userdata = _session_userdata(context)
    if userdata is None:
        return "Noted."

    progress = userdata.setdefault("lead_progress", {})
    if name:
        progress["name"] = name
    if email:
        normalized_email = normalize_spoken_email(email)
        if not is_valid_email(normalized_email):
            return (
                "The email hypothesis is invalid after normalization. Ask the caller "
                "to type it in chat, or collect it in two parts: first before @, then "
                "the domain."
            )
        progress["email"] = normalized_email
    if company:
        progress["company"] = company
    if reason_for_meet:
        progress["reason_for_meet"] = reason_for_meet

    _schedule_caller_checkpoint(userdata)
    if email:
        return (
            f"Email captured as {progress['email']}. Repeat it character by "
            f"character: {format_email_for_readback(progress['email'])}. Ask the "
            "caller to confirm it before relying on it."
        )
    return "Noted."


@function_tool
async def send_confirmed_lead_email_and_save(
    context: RunContext,
    name: str,
    email: str,
    company: str,
    reason_for_meet: str,
    email_confirmed: bool,
) -> str:
    """Send the Woice AI waitlist email, save the completed lead, and end the call.

    Only call this after the caller has clearly confirmed that the waitlist recap
    should be sent to their email address.
    """
    if not email_confirmed:
        return "Do not send email. Caller has not confirmed permission."

    email = normalize_spoken_email(email)
    if not is_valid_email(email):
        return (
            "Do not send email yet. The captured address looks malformed. Ask the "
            "caller to type it in the chat, or spell it out slowly, then read it "
            "back to confirm before trying again."
        )

    room_name = _get_room_name(context)
    caller_number = _get_caller_number_from_room(context)
    # Phone for telephony, ip:<addr> for web — so a returning web visitor is also
    # recognized. Falls back to the phone when both resolve to the same caller.
    caller_ref = _get_caller_ref_from_room(context) or caller_number
    lead_for_email = {
        "name": name,
        "email": email,
        "company": company,
        "reason_for_meet": reason_for_meet,
    }

    send_woice_waitlist_email(lead_for_email)
    saved = save_completed_lead_to_redis(
        room_name=room_name,
        caller_number=caller_number,
        name=name,
        email=email,
        company=company,
        reason_for_meet=reason_for_meet,
        email_confirmed=True,
        email_sent=True,
    )

    # Remember this caller (phone or web IP) so a return visit is greeted by name.
    # Marking the session complete stops the shutdown hook from writing a "partial".
    save_caller_memory(
        phone=caller_ref,
        status="completed",
        name=name,
        email=email,
        company=company,
        reason_for_meet=reason_for_meet,
        lead_id=saved["lead_id"],
    )
    userdata = _session_userdata(context)
    if userdata is not None:
        userdata["lead_completed"] = True

    speech = context.session.say(
        "Done. I have added you to the waitlist and sent the recap to your email. "
        f"Thanks for calling {company_name()}.",
        allow_interruptions=False,
        add_to_chat_ctx=True,
    )
    await speech.wait_for_playout()
    await _end_room_after_playout(context)

    return f"Email sent and lead saved. Lead ID: {saved['lead_id']}"
