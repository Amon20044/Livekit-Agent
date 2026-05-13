import pytest

from tools import _announce_search, search_ai_mode, search_latest_news


class _FakeSession:
    def __init__(self) -> None:
        self.instructions = []

    async def generate_reply(self, *, instructions: str) -> None:
        self.instructions.append(instructions)


class _FakeContext:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.playout_count = 0

    async def wait_for_playout(self) -> None:
        self.playout_count += 1


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

    assert len(context.session.instructions) == 1
    assert "yes sure" not in context.session.instructions[0].lower()
    assert context.playout_count == 1
