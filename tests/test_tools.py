import pytest

from tools import search_ai_mode, search_latest_news


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
