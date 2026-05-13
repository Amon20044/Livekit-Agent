from agent import AnchorVoiceAgent, _env_bool, _plugin_model
from tools import search_ai_mode, search_latest_news


def test_anchor_agent_is_search_focused_and_has_serpapi_tools() -> None:
    agent = AnchorVoiceAgent()

    assert "Anchor" in agent.instructions
    assert "latest news" in agent.instructions
    assert "search_ai_mode" in agent.instructions
    assert agent.tools == [search_latest_news, search_ai_mode]


def test_plugin_model_accepts_prefixed_and_legacy_values() -> None:
    assert _plugin_model("deepgram/nova-3-general", "deepgram") == "nova-3"
    assert _plugin_model("google/gemini-2.5-flash-lite", "google") == (
        "gemini-2.5-flash-lite"
    )
    assert _plugin_model("elevenlabs/eleven_flash_v2_5", "elevenlabs") == (
        "eleven_flash_v2_5"
    )


def test_env_bool(monkeypatch) -> None:
    monkeypatch.delenv("PREEMPTIVE_GENERATION", raising=False)
    assert _env_bool("PREEMPTIVE_GENERATION", True) is True

    monkeypatch.setenv("PREEMPTIVE_GENERATION", "false")
    assert _env_bool("PREEMPTIVE_GENERATION", True) is False

    monkeypatch.setenv("PREEMPTIVE_GENERATION", "yes")
    assert _env_bool("PREEMPTIVE_GENERATION", False) is True
