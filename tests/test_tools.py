import aiohttp
import pytest

from tools import (
    SerpApiError,
    _announce_search,
    _log_serpapi_client_error,
    _redact_url,
    search_ai_mode,
    search_latest_news,
)


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


def test_redact_url_removes_serpapi_key() -> None:
    url = "https://serpapi.com/search?engine=google_ai_mode&q=test&api_key=secret"

    assert _redact_url(url) == "https://serpapi.com/search?engine=google_ai_mode&q=test"


@pytest.mark.asyncio
async def test_search_ai_mode_retries_without_location_on_400(monkeypatch) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "secret")
    calls = []

    async def fake_serpapi_get(params, timeout_seconds=12):
        calls.append(params)
        if len(calls) == 1:
            raise SerpApiError(400, "location is not supported")
        return {
            "reconstructed_markdown": "Here are a few good options.",
            "references": [{"title": "Example", "source": "Local Guide"}],
        }

    monkeypatch.setattr("tools._serpapi_get", fake_serpapi_get)

    result = await search_ai_mode._func(
        _FakeContext(),
        "restaurants near Whitefield in Bangalore",
        location="Whitefield, Bangalore",
    )

    assert "Here are a few good options" in result
    assert calls[0]["location"] == "Whitefield, Bangalore"
    assert "location" not in calls[1]


@pytest.mark.asyncio
async def test_search_ai_mode_failure_does_not_return_secret(monkeypatch) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "secret")

    async def fake_serpapi_get(params, timeout_seconds=12):
        raise SerpApiError(400, "bad request")

    monkeypatch.setattr("tools._serpapi_get", fake_serpapi_get)

    result = await search_ai_mode._func(_FakeContext(), "query")

    assert "secret" not in result
    assert "API key was not exposed" in result


def test_client_error_logging_redacts_api_key(caplog) -> None:
    request_info = aiohttp.RequestInfo(
        url="https://serpapi.com/search?api_key=secret&q=test",
        method="GET",
        headers={},
        real_url="https://serpapi.com/search?api_key=secret&q=test",
    )
    exc = aiohttp.ClientResponseError(
        request_info=request_info,
        history=(),
        status=400,
        message="Bad Request",
    )

    _log_serpapi_client_error(exc)

    assert "secret" not in caplog.text
    assert "q=test" in caplog.text
