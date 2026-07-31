from app.models import RuntimeConfig


def test_runtime_config_parses_go_camel_case_contract() -> None:
    runtime = RuntimeConfig.from_payload(
        {
            "swarm": {
                "id": "swarm-1",
                "telephony": {
                    "phoneNumber": "+19388004249",
                    "outboundTrunkId": "ST_outbound",
                    "humanHandoffNumber": "+918126679138",
                },
                "graph": {
                    "entryNodeKey": "concierge",
                    "nodes": [
                        {
                            "key": "concierge",
                            "agentId": "agent-1",
                            "humanHandoffNumber": "+918126679138",
                            "instructionOverrides": "Keep it short.",
                            "metadata": {},
                        }
                    ],
                    "edges": [],
                },
            },
            "agents": [
                {
                    "id": "agent-1",
                    "name": "Concierge",
                    "instructions": {
                        "system": "Help the caller.",
                        "greeting": "Hello",
                        "guardrails": ["Verify facts"],
                        "fallback": "I cannot verify that.",
                    },
                    "model": {
                        "provider": "openai",
                        "name": "gpt-5.6-luna",
                        "temperature": 0.2,
                        "maxOutputTokens": 500,
                        "reasoningEffort": "none",
                    },
                    "voice": {
                        "language": "multi",
                        "sttProvider": "deepgram",
                        "sttModel": "nova-3",
                        "ttsProvider": "sarvam",
                        "ttsModel": "bulbul:v3",
                        "speaker": "shubh",
                        "pace": 1,
                        "allowInterruption": True,
                        "endCallAfterSec": 900,
                    },
                    "tools": [
                        {
                            "key": "handoff",
                            "description": "Move to next node",
                            "enabled": True,
                        }
                    ],
                    "capture": {
                        "fields": [
                            {
                                "key": "intent",
                                "label": "Intent",
                                "type": "string",
                                "description": "Reason",
                                "required": True,
                            }
                        ]
                    },
                }
            ],
            "call": {
                "participant": {"name": "Demo shopper"},
                "context": {"cart_value": 2499},
                "campaign": {
                    "language": "gu-IN",
                    "instructions": {
                        "objective": "Recover the cart",
                        "system": "Ask about the delivery concern.",
                        "greeting": "Namaste from StylMe",
                    },
                    "capture": {
                        "fields": [
                            {
                                "key": "checkout_interest",
                                "label": "Interest",
                                "type": "string",
                                "description": "Explicit checkout intent",
                                "required": True,
                            }
                        ]
                    },
                },
            },
            "credentials": {"sarvam": "runtime-secret"},
        }
    )
    assert runtime.swarm_id == "swarm-1"
    assert runtime.telephony.phone_number == "+19388004249"
    assert runtime.telephony.outbound_trunk_id == "ST_outbound"
    assert runtime.telephony.human_handoff_number == "+918126679138"
    assert runtime.graph.entry_node_key == "concierge"
    assert runtime.node("concierge").human_handoff_number == "+918126679138"
    assert runtime.human_handoff_number_for_node("concierge") == "+918126679138"
    assert runtime.agent_for_node("concierge").model.name == "gpt-5.6-luna"
    assert "Keep it short" in runtime.instructions_for_node("concierge")
    instructions = runtime.instructions_for_node("concierge")
    assert "untrusted factual data" in instructions
    assert "Demo shopper" in instructions
    assert "Recover the cart" in instructions
    assert runtime.voice_for_node("concierge").language == "gu-IN"
    assert runtime.greeting_for_node("concierge") == "Namaste from StylMe"
    assert runtime.capture_fields_for_node("concierge")[-1].key == "checkout_interest"
    assert runtime.credentials["sarvam"] == "runtime-secret"


def test_runtime_config_falls_back_to_legacy_swarm_handoff_number() -> None:
    runtime = RuntimeConfig.from_payload(
        {
            "swarm": {
                "id": "legacy-swarm",
                "telephony": {"humanHandoffNumber": "+918111111111"},
                "graph": {
                    "entryNodeKey": "human",
                    "nodes": [{"key": "human", "agentId": "agent-human"}],
                    "edges": [],
                },
            },
            "agents": [{"id": "agent-human", "name": "Human handoff"}],
        }
    )

    assert runtime.human_handoff_number_for_node("human") == "+918111111111"
