import asyncio
from typing import Any

from livekit.agents import llm

from app.agent_session import (
    VoiceState,
    apply_opt_out,
    brief_answered_human_and_connect,
    configured_tools_for_node,
    copy_handoff_chat_context,
    perform_warm_transfer,
    prepare_handoff,
)
from app.models import RuntimeConfig


def _runtime(*, tools: list[str], has_handoff_edge: bool = True) -> RuntimeConfig:
    return RuntimeConfig.from_payload(
        {
            "swarm": {
                "id": "swarm-1",
                "graph": {
                    "entryNodeKey": "concierge",
                    "nodes": [
                        {"key": "concierge", "agentId": "agent-1"},
                        {"key": "specialist", "agentId": "agent-2"},
                    ],
                    "edges": (
                        [
                            {
                                "from": "concierge",
                                "to": "specialist",
                                "priority": 1,
                                "condition": {
                                    "field": "intent",
                                    "operator": "exists",
                                },
                            }
                        ]
                        if has_handoff_edge
                        else []
                    ),
                },
            },
            "agents": [
                {
                    "id": "agent-1",
                    "name": "Concierge",
                    "instructions": {"system": "Help the caller."},
                    "tools": [
                        {"key": key, "enabled": True, "description": key}
                        for key in tools
                    ],
                    "capture": {
                        "fields": [
                            {
                                "key": "intent",
                                "label": "Intent",
                                "description": "Caller intent",
                                "required": True,
                            }
                        ]
                    },
                },
                {
                    "id": "agent-2",
                    "name": "Specialist",
                    "instructions": {"system": "Handle the request."},
                },
            ],
        }
    )


def _state(runtime: RuntimeConfig) -> VoiceState:
    return VoiceState(
        runtime=runtime,
        control=object(),  # type: ignore[arg-type]
        call_id="call-1",
        current_node_key="concierge",
        direction="inbound",
        room_name="room-1",
    )


def _tool_names(tools: list[Any]) -> list[str]:
    return [tool.info.name for tool in tools]


def test_configured_tools_are_unique_and_respect_enabled_flags() -> None:
    tools = configured_tools_for_node(
        _state(_runtime(tools=["search_catalog", "handoff", "end_call"])),
        "concierge",
    )

    names = _tool_names(tools)
    assert names == [
        "capture_call_field",
        "search_catalog",
        "handoff_to_next_agent",
        "end_call",
    ]
    assert len(names) == len(set(names))
    llm.ToolContext(tools)


def test_handoff_tool_requires_an_outgoing_edge() -> None:
    tools = configured_tools_for_node(
        _state(_runtime(tools=["handoff"], has_handoff_edge=False)),
        "concierge",
    )

    assert _tool_names(tools) == ["capture_call_field"]


def test_disabled_tools_are_not_auto_registered() -> None:
    tools = configured_tools_for_node(_state(_runtime(tools=[])), "concierge")

    assert _tool_names(tools) == ["capture_call_field"]


def test_order_and_callback_tools_are_only_registered_when_enabled() -> None:
    tools = configured_tools_for_node(
        _state(_runtime(tools=["lookup_order", "capture_callback"])), "concierge"
    )

    assert _tool_names(tools) == [
        "capture_call_field",
        "lookup_order",
        "capture_callback",
    ]


def test_record_opt_out_is_registered_and_persists_an_explicit_request() -> None:
    state = _state(_runtime(tools=["record_opt_out"]))

    assert _tool_names(configured_tools_for_node(state, "concierge")) == [
        "capture_call_field",
        "record_opt_out",
    ]
    assert apply_opt_out(state, confirmed=False, reason="maybe later") == (
        "Opt-out not recorded because explicit confirmation is required."
    )
    assert "opt_out" not in state.captured

    assert apply_opt_out(state, confirmed=True, reason="Do not call again") == (
        "Do-not-call preference recorded for this call."
    )
    assert state.captured["opt_out"] is True
    assert state.captured["opt_out_reason"] == "Do not call again"


def test_warm_transfer_tool_is_only_registered_when_enabled() -> None:
    tools = configured_tools_for_node(
        _state(_runtime(tools=["warm_transfer"])), "concierge"
    )

    assert _tool_names(tools) == ["capture_call_field", "warm_transfer_to_human"]


async def test_warm_transfer_dials_admin_number_through_the_managed_trunk() -> None:
    runtime = _runtime(tools=["warm_transfer"])
    runtime.telephony.phone_number = "+19388004249"
    runtime.telephony.outbound_trunk_id = "ST_outbound"
    runtime.node("concierge").human_handoff_number = "+918126679138"
    state = _state(runtime)
    captured: dict[str, Any] = {}

    class FakeWarmTransferTask:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def __await__(self):
            async def complete():
                return type("Result", (), {"human_agent_identity": "human-agent-sip"})()

            return complete().__await__()

    result = await perform_warm_transfer(
        state,
        chat_ctx="safe-chat-context",
        summary="Caller needs a verified refund update.",
        task_factory=FakeWarmTransferTask,
    )

    assert result.human_agent_identity == "human-agent-sip"
    assert captured["sip_call_to"] == "+918126679138"
    assert captured["sip_trunk_id"] == "ST_outbound"
    assert captured["sip_number"] == "+19388004249"
    assert captured["chat_ctx"] == "safe-chat-context"
    assert captured["ringing_timeout"] == 30.0
    assert captured["briefing_timeout"] == 12.0
    assert "Connecting you now" in captured["answer_briefing"]


async def test_warm_transfer_uses_the_current_hil_node_destination() -> None:
    runtime = _runtime(tools=["warm_transfer"])
    runtime.telephony.phone_number = "+19388004249"
    runtime.telephony.outbound_trunk_id = "ST_outbound"
    runtime.telephony.human_handoff_number = "+918000000000"
    runtime.node("concierge").human_handoff_number = "+919999999999"
    state = _state(runtime)
    captured: dict[str, Any] = {}

    class FakeWarmTransferTask:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def __await__(self):
            async def complete():
                return type("Result", (), {"human_agent_identity": "human-agent-sip"})()

            return complete().__await__()

    await perform_warm_transfer(
        state,
        chat_ctx="safe-chat-context",
        summary="Caller asked for a human.",
        task_factory=FakeWarmTransferTask,
    )

    assert captured["sip_call_to"] == "+919999999999"


async def test_answered_human_is_briefed_then_connected_without_second_confirmation() -> (
    None
):
    events: list[str] = []

    class FakeSpeech:
        async def wait_for_playout(self) -> None:
            events.append("briefed")

    class FakeHumanSession:
        def say(self, text: str, *, allow_interruptions: bool) -> FakeSpeech:
            assert "Connecting you now" in text
            assert allow_interruptions is False
            events.append("briefing_started")
            return FakeSpeech()

    async def connect() -> None:
        events.append("connected")

    await brief_answered_human_and_connect(
        FakeHumanSession(),
        briefing="The caller needs order support. Connecting you now.",
        connect=connect,
    )

    assert events == ["briefing_started", "briefed", "connected"]


async def test_answered_human_connects_even_when_briefing_times_out() -> None:
    connected = False

    class SlowSpeech:
        async def wait_for_playout(self) -> None:
            await asyncio.sleep(1)

    class SlowHumanSession:
        def say(self, _text: str, *, allow_interruptions: bool) -> SlowSpeech:
            assert allow_interruptions is False
            return SlowSpeech()

    async def connect() -> None:
        nonlocal connected
        connected = True

    await brief_answered_human_and_connect(
        SlowHumanSession(),
        briefing="Connecting you now.",
        connect=connect,
        briefing_timeout=0.01,
    )

    assert connected is True


async def test_warm_transfer_fails_before_dial_when_admin_number_is_missing() -> None:
    state = _state(_runtime(tools=["warm_transfer"]))

    try:
        await perform_warm_transfer(
            state,
            chat_ctx="safe-chat-context",
            summary="Needs a person.",
        )
    except RuntimeError as exc:
        assert "human handoff number" in str(exc)
    else:
        raise AssertionError("missing human handoff configuration must fail closed")


def test_prepare_handoff_uses_explicit_route_for_multi_layer_workflows() -> None:
    runtime = RuntimeConfig.from_payload(
        {
            "swarm": {
                "id": "swarm-1",
                "graph": {
                    "entryNodeKey": "receptionist",
                    "nodes": [
                        {"key": "receptionist", "agentId": "agent-1"},
                        {"key": "orchestrator", "agentId": "agent-2"},
                        {"key": "shopping", "agentId": "agent-3"},
                    ],
                    "edges": [
                        {
                            "from": "receptionist",
                            "to": "orchestrator",
                            "priority": 100,
                            "condition": {
                                "field": "handoff_route",
                                "operator": "eq",
                                "value": "orchestrator",
                            },
                        },
                        {
                            "from": "orchestrator",
                            "to": "shopping",
                            "priority": 100,
                            "condition": {
                                "field": "handoff_route",
                                "operator": "eq",
                                "value": "shopping",
                            },
                        },
                    ],
                },
            },
            "agents": [
                {
                    "id": "agent-1",
                    "name": "Receptionist",
                    "instructions": {"system": "Intake."},
                },
                {
                    "id": "agent-2",
                    "name": "Router",
                    "instructions": {"system": "Route."},
                },
                {
                    "id": "agent-3",
                    "name": "Shopping",
                    "instructions": {"system": "Shop."},
                },
            ],
        }
    )
    state = _state(runtime)
    state.current_node_key = "receptionist"

    first = prepare_handoff(state, reason="festive outfit", route="orchestrator")
    assert first is not None and first.to_node == "orchestrator"
    assert state.captured["handoff_route"] == "orchestrator"
    assert state.captured["intent"] == "festive outfit"

    state.current_node_key = "orchestrator"
    second = prepare_handoff(state, reason="shopping request", route="shopping")
    assert second is not None and second.to_node == "shopping"
    assert state.captured["handoff_route"] == "shopping"
    assert state.captured["intent"] == "festive outfit"


def test_handoff_context_excludes_prior_agent_instructions_and_config_updates() -> None:
    class FakeChatContext:
        def __init__(self) -> None:
            self.kwargs: dict[str, bool] = {}

        def copy(self, **kwargs: bool) -> str:
            self.kwargs = kwargs
            return "safe-context"

    chat_ctx = FakeChatContext()

    assert copy_handoff_chat_context(chat_ctx) == "safe-context"
    assert chat_ctx.kwargs == {
        "exclude_instructions": True,
        "exclude_config_update": True,
    }
