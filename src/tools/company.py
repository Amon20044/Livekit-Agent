import json
import logging
import os
import re
import smtplib
import ssl
import time
import uuid
from email.message import EmailMessage
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
    (r"\bat the rate(?: of)?\b", "@"),
    (r"\bunderscore\b", "_"),
    (r"\b(?:dash|hyphen|minus)\b", "-"),
    (r"\bplus\b", "+"),
    (r"\bdot\b", "."),
    (r"\bat\b", "@"),
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

    text = re.sub(r"\s+", "", text)
    # Collapse separators that spacing/substitution may have duplicated.
    text = re.sub(r"@{2,}", "@", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip(".")


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


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


def _redis_client() -> Redis:
    redis_url = _clean_env("REDIS_URL") or "redis://localhost:6379/0"
    return Redis.from_url(redis_url, decode_responses=True)


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
        f"dreamlaunch:lead:{lead_id}",
        json.dumps(lead),
        ex=ttl_seconds,
    )
    redis.set(
        f"dreamlaunch:room:{room_name}:lead",
        lead_id,
        ex=ttl_seconds,
    )

    return lead


def _required_env(name: str) -> str:
    value = _clean_env(name)
    if not value:
        raise RuntimeError(f"{name} is required to send DreamLaunch recap emails")
    return value


def _smtp_config() -> dict[str, str | int]:
    google_app_email = _clean_env("GOOGLE_APP_EMAIL")
    google_app_pass = _clean_env("GOOGLE_APP_PASS")

    smtp_host = _clean_env("SMTP_HOST") or "smtp.gmail.com"
    smtp_port = int(_clean_env("SMTP_PORT") or "587")
    smtp_username = google_app_email or _required_env("SMTP_USERNAME")
    smtp_password = google_app_pass or _required_env("SMTP_PASSWORD")
    sender = _clean_env("SMTP_FROM") or smtp_username
    reply_to = _clean_env("DREAMLAUNCH_REPLY_TO") or sender

    return {
        "host": smtp_host,
        "port": smtp_port,
        "username": smtp_username,
        "password": smtp_password,
        "sender": sender,
        "reply_to": reply_to,
    }


def company_name() -> str:
    return _clean_env("COMPANY_NAME") or "DreamLaunch Studio"


def company_website() -> str:
    return _clean_env("COMPANY_WEBSITE") or "https://dreamlaunch.studio"


def _email_body(lead: dict[str, str | None]) -> str:
    company = company_name()
    website = company_website()
    booking_url = _clean_env("DREAMLAUNCH_BOOKING_URL") or website
    company_line = f"Company: {lead.get('company') or 'Not provided'}"
    return f"""Hi {lead["name"]},

Thanks for calling {company}. Here is the brief I captured:

Name: {lead["name"]}
Email: {lead["email"]}
{company_line}
Reason for meeting: {lead["reason_for_meet"]}

Next step:
Book a strategy call here: {booking_url}

We will use this brief to prepare the conversation and keep the first call focused.

{company}
{website}
"""


def send_dreamlaunch_recap_email(lead: dict[str, str | None]) -> None:
    smtp_config = _smtp_config()

    message = EmailMessage()
    message["Subject"] = f"Your {company_name()} brief and booking link"
    message["From"] = str(smtp_config["sender"])
    message["To"] = str(lead["email"])
    message["Reply-To"] = str(smtp_config["reply_to"])
    message.set_content(_email_body(lead))

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


def _get_caller_number_from_room(context: RunContext) -> str | None:
    room = getattr(context.session, "room", None)
    participants = getattr(room, "remote_participants", {}) or {}

    for participant in participants.values():
        identity = getattr(participant, "identity", "") or ""
        if identity.startswith("sip_"):
            return identity.removeprefix("sip_")
        if identity.startswith("+"):
            return identity

    return None


async def _end_room_after_playout(context: RunContext) -> None:
    job_ctx = get_job_context()
    if job_ctx is not None:
        await job_ctx.delete_room()
        return

    shutdown = getattr(context.session, "shutdown", None)
    if callable(shutdown):
        shutdown(drain=True)


@function_tool
async def send_confirmed_lead_email_and_save(
    context: RunContext,
    name: str,
    email: str,
    company: str,
    reason_for_meet: str,
    email_confirmed: bool,
) -> str:
    """Send the DreamLaunch recap email, save the completed lead, and end the call.

    Only call this after the caller has clearly confirmed that the brief and
    booking link should be sent to their email address.
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
    lead_for_email = {
        "name": name,
        "email": email,
        "company": company,
        "reason_for_meet": reason_for_meet,
    }

    send_dreamlaunch_recap_email(lead_for_email)
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

    speech = context.session.say(
        "Done. I have sent the recap and booking link to your email. "
        f"Thanks for calling {company_name()}.",
        allow_interruptions=False,
        add_to_chat_ctx=True,
    )
    await speech.wait_for_playout()
    await _end_room_after_playout(context)

    return f"Email sent and lead saved. Lead ID: {saved['lead_id']}"
