import pytest

from tools import _announce_search, search_ai_mode, search_latest_news


class _FakeSession:
    def __init__(self) -> None:
        self.said = []

    def say(
        self,
        text: str,
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
    ) -> "_FakeSpeechHandle":
        self.said.append(
            {
                "text": text,
                "allow_interruptions": allow_interruptions,
                "add_to_chat_ctx": add_to_chat_ctx,
            }
        )
        return _FakeSpeechHandle()


class _FakeSpeechHandle:
    def __init__(self) -> None:
        self.playout_count = 0

    async def wait_for_playout(self) -> None:
        self.playout_count += 1


class _FakeContext:
    def __init__(self) -> None:
        self.session = _FakeSession()


@pytest.mark.asyncio
async def test_search_latest_news_reports_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    result = await search_latest_news._func(None, "latest AI news")

    assert "SERPAPI_API_KEY is not configured" in result


@pytest.mark.asyncio
async def test_search_ai_mode_reports_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    result = await search_ai_mode._func(None, "best lightweight laptop")

    assert "SERPAPI_API_KEY is not configured" in result


@pytest.mark.asyncio
async def test_search_announcement_only_speaks_once_per_context() -> None:
    context = _FakeContext()

    await _announce_search(context)
    await _announce_search(context)

    assert context.session.said == [
        {
            "text": "I'll check that now.",
            "allow_interruptions": False,
            "add_to_chat_ctx": False,
        }
    ]
