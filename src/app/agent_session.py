from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RunContext,
    function_tool,
    get_job_context,
    llm,
    room_io,
)
from livekit.agents.beta.workflows import WarmTransferTask, WorkflowInstructions
from livekit.plugins import silero

from app.control_plane import ControlPlaneClient, parse_job_metadata
from app.graph import choose_transition, graph_is_acyclic
from app.models import RuntimeConfig
from inferences.llm import build_llm
from inferences.stt import build_stt
from inferences.tts import build_tts
from inferences.turn import build_turn_handling
from settings import (
    CONTROL_PLANE_URL,
    DEFAULT_INBOUND_SWARM_ID,
    DEFAULT_OUTBOUND_SWARM_ID,
    INTERNAL_API_KEY,
    PYTHON_API_BASE_URL,
    validate_provider_credentials,
    validate_runtime_settings,
)

logger = logging.getLogger("stylme.voice")

_PHONE_LAST4 = re.compile(r"^[0-9]{4}$")


async def brief_answered_human_and_connect(
    human_session: Any,
    *,
    briefing: str,
    connect: Any,
    briefing_timeout: float = 12.0,
) -> None:
    """Brief an answered support line, then bridge without another voice gate."""
    try:
        async with asyncio.timeout(briefing_timeout):
            speech = human_session.say(briefing, allow_interruptions=False)
            await speech.wait_for_playout()
    except TimeoutError:
        logger.warning("human transfer briefing timed out; connecting immediately")
    except Exception as exc:
        logger.warning(
            "human transfer briefing failed; connecting immediately: %s", exc
        )
    await connect()


class AnsweredWarmTransferTask(WarmTransferTask):
    """Treat an answered support call as acceptance and bridge deterministically."""

    def __init__(
        self,
        *args: Any,
        answer_briefing: str,
        briefing_timeout: float = 12.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._answer_briefing = answer_briefing
        self._briefing_timeout = briefing_timeout

    async def on_enter(self) -> None:
        await super().on_enter()
        human_session = self._human_agent_sess
        if human_session is None or self.done():
            return

        logger.info("human support answered; briefing and bridging automatically")

        async def connect_if_active() -> None:
            if self.done():
                return
            await self.connect_to_caller()

        await brief_answered_human_and_connect(
            human_session,
            briefing=self._answer_briefing,
            connect=connect_if_active,
            briefing_timeout=self._briefing_timeout,
        )


@dataclass(slots=True)
class VoiceState:
    runtime: RuntimeConfig
    control: ControlPlaneClient
    call_id: str
    current_node_key: str
    direction: str
    room_name: str
    captured: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    greeted: bool = False


@function_tool()
async def capture_call_field(
    context: RunContext[VoiceState], key: str, value: str
) -> str:
    """Capture one explicit caller answer for the final disposition.

    Use only keys named in the current capture contract. Never infer a value.
    """
    state = context.userdata
    allowed = {
        item.key
        for item in state.runtime.capture_fields_for_node(state.current_node_key)
    }
    if key not in allowed:
        return f"'{key}' is not in this agent's capture contract."
    state.captured[key] = value.strip()
    return f"Captured {key}."


@function_tool()
async def search_catalog(context: RunContext[VoiceState], query: str) -> str:
    """Search real StylMe products for the caller's fashion request.

    Use this before stating product, price, stock, or delivery facts.
    """
    state = context.userdata
    try:
        result = await state.control.search_catalog(PYTHON_API_BASE_URL, query)
    except RuntimeError as exc:
        logger.warning("catalog tool failed: %s", exc)
        return "StylMe catalogue search is temporarily unavailable. Do not invent products or prices."
    products = []
    for item in (result.get("items") or [])[:5]:
        if not isinstance(item, dict):
            continue
        products.append(
            {
                "id": item.get("id") or item.get("_id"),
                "title": item.get("title") or item.get("name"),
                "brand": item.get("brand"),
                "pricePaise": item.get("pricePaise") or item.get("sale_price_paise"),
                "swoopStylEligible": item.get("swoopStylEligible"),
            }
        )
    return json.dumps(
        {"products": products, "total": result.get("total", len(products))},
        ensure_ascii=False,
    )


@function_tool()
async def lookup_order(
    context: RunContext[VoiceState], order_number: str, phone_last4: str
) -> str:
    """Verify an order with its number and the account phone's last four digits.

    Args:
        order_number: The exact order number stated by the caller.
        phone_last4: Exactly the final four digits of the account phone number.
    """
    normalized_order = order_number.strip()
    normalized_last4 = re.sub(r"\D", "", phone_last4)
    if not normalized_order or not _PHONE_LAST4.fullmatch(normalized_last4):
        return "Ask for the order number and exactly the last four phone digits."
    state = context.userdata
    try:
        result = await state.control.lookup_order(
            PYTHON_API_BASE_URL, normalized_order, normalized_last4
        )
    except RuntimeError as exc:
        logger.warning("order lookup failed: %s", exc)
        return "The order could not be verified. Do not reveal or invent any order details."
    state.captured["order_number"] = normalized_order
    state.captured["phone_last4"] = normalized_last4
    state.captured["order_verified"] = True
    return json.dumps(result, ensure_ascii=False)


@function_tool()
async def capture_callback(
    context: RunContext[VoiceState],
    confirmed: bool,
    preferred_time: str,
    summary: str,
) -> str:
    """Record a human-support callback after the caller explicitly confirms it.

    Args:
        confirmed: True only after an explicit yes from the caller.
        preferred_time: The caller's preferred callback time or "any time".
        summary: A concise factual summary for the human support team.
    """
    if not confirmed:
        return "Callback not recorded because explicit confirmation is still required."
    state = context.userdata
    state.captured["callback_requested"] = True
    state.captured["preferred_callback_at"] = preferred_time.strip() or "any time"
    state.captured["handoff_summary"] = summary.strip()
    return "Callback request recorded in this call's support disposition."


def apply_opt_out(state: VoiceState, *, confirmed: bool, reason: str) -> str:
    """Persist only an explicit do-not-call request in the call disposition."""
    if not confirmed:
        return "Opt-out not recorded because explicit confirmation is required."
    state.captured["opt_out"] = True
    state.captured["opt_out_reason"] = reason.strip() or "Explicit do-not-call request"
    state.captured["outcome"] = "opt-out"
    return "Do-not-call preference recorded for this call."


@function_tool()
async def record_opt_out(
    context: RunContext[VoiceState], confirmed: bool, reason: str
) -> str:
    """Record an explicit request not to receive future outbound calls.

    Args:
        confirmed: True only after the caller clearly asks not to be called again.
        reason: The caller's own concise opt-out wording or reason.
    """
    return apply_opt_out(context.userdata, confirmed=confirmed, reason=reason)


async def perform_warm_transfer(
    state: VoiceState,
    *,
    chat_ctx: Any,
    summary: str,
    task_factory: Any = AnsweredWarmTransferTask,
) -> Any:
    """Run LiveKit's documented warm-transfer workflow with admin configuration."""
    telephony = state.runtime.telephony
    human_handoff_number = state.runtime.human_handoff_number_for_node(
        state.current_node_key
    )
    if not human_handoff_number:
        raise RuntimeError(
            "The current HIL node has no human handoff number configured."
        )
    if not telephony.outbound_trunk_id:
        raise RuntimeError("The swarm has no managed outbound trunk for human handoff.")
    instructions = WorkflowInstructions(
        extra=(
            "You are the StylMe human-support transfer assistant. Give the human "
            "a short factual summary, say that the caller is waiting, and connect "
            "them only after the human confirms they are ready. Escalation summary: "
            + summary.strip()
        )
    )
    return await task_factory(
        sip_call_to=human_handoff_number,
        sip_trunk_id=telephony.outbound_trunk_id,
        sip_number=telephony.phone_number,
        chat_ctx=chat_ctx,
        instructions=instructions,
        ringing_timeout=30.0,
        answer_briefing=(
            "This is a StylMe support transfer. The caller is waiting. "
            f"Summary: {summary.strip()[:280] or 'The caller requested live support.'} "
            "Connecting you now."
        ),
        briefing_timeout=12.0,
    )


@function_tool()
async def warm_transfer_to_human(context: RunContext[VoiceState], summary: str) -> str:
    """Call the admin-configured human and connect them to the caller's room.

    Args:
        summary: A concise factual summary of the caller's need and verified context.
    """
    state = context.userdata
    factual_summary = summary.strip() or str(state.captured.get("intent") or "")
    state.captured["handoff_summary"] = factual_summary
    try:
        result = await perform_warm_transfer(
            state,
            chat_ctx=copy_handoff_chat_context(context.session.current_agent.chat_ctx),
            summary=factual_summary,
        )
    except Exception as exc:
        logger.warning("warm transfer failed: %s", exc)
        state.captured["human_handoff_status"] = "unavailable"
        return (
            "The live human transfer is unavailable. Tell the caller clearly, then "
            "offer to record a callback request without claiming anyone has joined."
        )

    state.captured["human_handoff_status"] = "connected"
    state.captured["human_agent_identity"] = result.human_agent_identity
    spoken_summary = factual_summary[:240] or "The caller requested live support."
    speech = context.session.say(
        "Your StylMe support specialist is now connected. "
        f"For context: {spoken_summary} I'll leave you both to continue.",
        allow_interruptions=False,
    )
    await speech.wait_for_playout()
    await context.session.shutdown(drain=True)
    return "Human support connected; the AI assistant left the room."


def prepare_handoff(state: VoiceState, *, reason: str, route: str = "") -> Any:
    """Apply one explicit route and resolve the matching outgoing graph edge."""
    normalized_reason = reason.strip()
    normalized_route = re.sub(r"[^a-z0-9_]+", "_", route.strip().casefold()).strip("_")
    state.captured.setdefault("intent", normalized_reason)
    if not normalized_route:
        return choose_transition(
            state.runtime.graph, state.current_node_key, state.captured
        )

    previous_route = state.captured.get("handoff_route")
    state.captured["handoff_route"] = normalized_route
    edge = choose_transition(
        state.runtime.graph, state.current_node_key, state.captured
    )
    if edge is None:
        if previous_route is None:
            state.captured.pop("handoff_route", None)
        else:
            state.captured["handoff_route"] = previous_route
    return edge


def copy_handoff_chat_context(chat_ctx: Any) -> Any:
    """Carry conversation turns forward without prior prompts or config records."""
    return chat_ctx.copy(
        exclude_instructions=True,
        exclude_config_update=True,
    )


@function_tool()
async def handoff_to_next_agent(
    context: RunContext[VoiceState], reason: str, route: str = ""
) -> tuple[Agent, str] | str:
    """Hand off only when captured data matches an allowed outgoing DAG edge.

    Args:
        reason: A concise factual summary of why the handoff is needed.
        route: The exact outgoing route named in the current agent instructions.
    """
    state = context.userdata
    node_key = state.current_node_key
    edge = prepare_handoff(state, reason=reason, route=route)
    if edge is None:
        return "No configured handoff condition matches. Continue in the current role or use the verified fallback."
    await state.control.record_handoff(
        state.call_id,
        from_node=node_key,
        to_node=edge.to_node,
        reason=reason,
        captured=state.captured,
    )
    state.current_node_key = edge.to_node
    message = (
        edge.handoff_message
        or f"Transferring to {state.runtime.agent_for_node(edge.to_node).name}."
    )
    return StylMeVoiceAgent(
        state,
        edge.to_node,
        chat_ctx=copy_handoff_chat_context(context.session.current_agent.chat_ctx),
    ), message


@function_tool()
async def end_call(context: RunContext[VoiceState], reason: str) -> str:
    """End the room after a clear goodbye, opt-out, or terminal workflow state."""
    context.userdata.captured.setdefault("end_reason", reason.strip())
    speech = context.session.say(
        "Thank you for your time. Goodbye.", allow_interruptions=False
    )
    await speech.wait_for_playout()
    job = get_job_context()
    if job is not None:
        await job.delete_room()
    return "Call ended."


def configured_tools_for_node(
    state: VoiceState, node_key: str
) -> list[llm.FunctionTool]:
    """Return only the explicitly enabled tools for one graph node."""
    config = state.runtime.agent_for_node(node_key)
    tools = [capture_call_field]
    if config.tool_enabled("search_catalog"):
        tools.append(search_catalog)
    if config.tool_enabled("lookup_order"):
        tools.append(lookup_order)
    if config.tool_enabled("capture_callback"):
        tools.append(capture_callback)
    if config.tool_enabled("warm_transfer"):
        tools.append(warm_transfer_to_human)
    if config.tool_enabled("record_opt_out"):
        tools.append(record_opt_out)
    if config.tool_enabled("handoff") and any(
        edge.from_node == node_key for edge in state.runtime.graph.edges
    ):
        tools.append(handoff_to_next_agent)
    if config.tool_enabled("end_call"):
        tools.append(end_call)
    return tools


class StylMeVoiceAgent(Agent):
    def __init__(self, state: VoiceState, node_key: str, *, chat_ctx=None) -> None:
        self.state = state
        self.node_key = node_key
        config = state.runtime.agent_for_node(node_key)
        voice = state.runtime.voice_for_node(node_key)
        super().__init__(
            instructions=state.runtime.instructions_for_node(node_key),
            tools=configured_tools_for_node(state, node_key),
            llm=build_llm(config.model, state.runtime.credentials.get("openai", "")),
            stt=build_stt(voice, state.runtime.credentials.get("deepgram", "")),
            tts=build_tts(voice, state.runtime.credentials.get("sarvam", "")),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        config = self.state.runtime.agent_for_node(self.node_key)
        node = self.state.runtime.node(self.node_key)
        self.state.current_node_key = self.node_key
        if not self.state.greeted:
            self.state.greeted = True
            greeting = self.state.runtime.greeting_for_node(self.node_key)
            instruction = f"Say this greeting naturally in the caller's language, without adding claims: {greeting}"
        elif node.metadata.get("silentHandoff") is True:
            instruction = (
                "Do not speak to the caller. Immediately use the configured handoff "
                "tool to route from the existing conversation context."
            )
        elif config.tool_enabled("warm_transfer"):
            instruction = (
                "Tell the caller in one short sentence that you are connecting live "
                "human support, then immediately call warm_transfer_to_human with a "
                "concise factual summary from the existing conversation."
            )
        else:
            instruction = f"Briefly introduce your role as {config.name}, acknowledge the handoff context, then continue."
        await self.session.generate_reply(
            instructions=instruction, allow_interruptions=False
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.45,
        prefix_padding_duration=0.35,
        activation_threshold=0.5,
        force_cpu=True,
    )


async def entrypoint(ctx: JobContext) -> None:
    validate_runtime_settings()
    await ctx.connect()
    control = ControlPlaneClient(CONTROL_PLANE_URL, INTERNAL_API_KEY)
    metadata = parse_job_metadata(getattr(ctx.job, "metadata", ""))
    swarm_id = metadata.swarm_id or (
        DEFAULT_OUTBOUND_SWARM_ID
        if metadata.direction == "outbound"
        else DEFAULT_INBOUND_SWARM_ID
    )
    call_id = metadata.call_id
    if metadata.direction == "inbound" or not call_id:
        caller, dialed = _sip_numbers(ctx.room)
        created = await control.create_inbound_call(
            swarm_id=swarm_id, room=ctx.room.name, caller=caller, dialed=dialed
        )
        call_id = str(created.get("id") or "")
    runtime = await control.runtime(swarm_id, call_id)
    validate_provider_credentials(runtime.credentials)
    if not graph_is_acyclic(runtime.graph):
        raise RuntimeError("The selected agent swarm is not a valid DAG")
    state = VoiceState(
        runtime=runtime,
        control=control,
        call_id=call_id,
        current_node_key=runtime.graph.entry_node_key,
        direction=metadata.direction,
        room_name=ctx.room.name,
    )
    initial = runtime.agent_for_node(runtime.graph.entry_node_key)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        turn_handling=build_turn_handling(),
        userdata=state,
    )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        item = event.item
        if not isinstance(item, llm.ChatMessage) or item.role not in {
            "user",
            "assistant",
        }:
            return
        text = item.text_content.strip()
        if not text:
            return
        state.transcript.append(
            {
                "role": item.role,
                "agentId": runtime.agent_for_node(state.current_node_key).id
                if item.role == "assistant"
                else "",
                "text": text,
                "createdAt": datetime.now(UTC).isoformat(),
            }
        )

    async def finalize() -> None:
        timeout_task.cancel()
        with suppress(asyncio.CancelledError):
            await timeout_task
        if not state.call_id:
            return
        try:
            await control.complete_call(state.call_id, state.transcript)
        except Exception:
            logger.exception("failed to persist post-call transcript/disposition")

    timeout_task = asyncio.create_task(
        _enforce_call_limit(ctx, session, initial.voice.end_call_after_sec)
    )
    ctx.add_shutdown_callback(finalize)
    logger.info(
        "starting StylMe call call_id=%s swarm_id=%s direction=%s entry=%s",
        call_id,
        swarm_id,
        metadata.direction,
        runtime.graph.entry_node_key,
    )
    await session.start(
        room=ctx.room,
        agent=StylMeVoiceAgent(state, runtime.graph.entry_node_key),
        room_options=room_io.RoomOptions(audio_input=True),
    )


async def _enforce_call_limit(
    ctx: JobContext, session: AgentSession, limit_seconds: int
) -> None:
    await asyncio.sleep(max(30, limit_seconds))
    speech = session.say(
        "We have reached the configured call time limit. Thank you for calling StylMe. Goodbye.",
        allow_interruptions=False,
    )
    await speech.wait_for_playout()
    await ctx.delete_room()


def _sip_numbers(room) -> tuple[str, str]:
    caller = ""
    dialed = ""
    for participant in (getattr(room, "remote_participants", {}) or {}).values():
        attributes = getattr(participant, "attributes", {}) or {}
        identity = getattr(participant, "identity", "") or ""
        caller = caller or str(
            attributes.get("sip.phoneNumber")
            or attributes.get("sip.from")
            or (
                identity.removeprefix("sip_")
                if identity.startswith("sip_")
                else identity
                if identity.startswith("+")
                else ""
            )
        )
        dialed = dialed or str(
            attributes.get("sip.trunkPhoneNumber") or attributes.get("sip.to") or ""
        )
    return caller, dialed
