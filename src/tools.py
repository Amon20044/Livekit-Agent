import logging
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from livekit.agents import RunContext, function_tool

logger = logging.getLogger(__name__)
SERPAPI_URL = "https://serpapi.com/search"


def _clean_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None

async def _serpapi_get(params: dict[str, str], timeout_seconds: int = 12) -> dict:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with ( 
        aiohttp.ClientSession(timeout=timeout) as session,
        session.get(SERPAPI_URL, params=params) as response,
    ):
        payload = await response.json(content_type=None)
        if response.status >= 400:
            raise SerpApiError(
                response.status,
                payload.get("error") or response.reason or "SerpApi request failed",
            )
        return payload


class SerpApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"SerpApi returned {status}: {message}")


def _redact_url(url: str) -> str:
    parsed = urlsplit(str(url))
    query_parts = [
        part for part in parsed.query.split("&") if not part.startswith("api_key=")
    ]
    return urlunsplit(parsed._replace(query="&".join(query_parts)))


def _log_serpapi_client_error(exc: aiohttp.ClientError) -> None:
    url = getattr(getattr(exc, "request_info", None), "real_url", "")
    logger.warning(
        "SerpApi request failed: status=%s url=%s",
        getattr(exc, "status", "unknown"),
        _redact_url(str(url)) if url else "unknown",
    )


def _source_name(source: Any) -> str | None:
    if isinstance(source, dict):
        return source.get("name")
    if isinstance(source, str):
        return source
    return None


def _google_country_code(location: str) -> str:
    normalized = location.strip().lower()
    if "india" in normalized or normalized in {"in", "bharat"}:
        return "in"
    return "us"


@function_tool
async def search_latest_news(
    context: RunContext, query: str, location: str = "United States"
) -> str:
    """Search recent news with SerpApi for current information.

    Use this when the user asks about latest news, recent events, current facts,
    or anything that could have changed recently.
    """
    api_key = _clean_env("SERPAPI_API_KEY")
    if not api_key:
        return "I can't search live news yet because SERPAPI_API_KEY is not configured."

    params = {
        "engine": "google_news",
        "q": query,
        "gl": "IN",
        "hl": "en",
        "location": location or "India",
        "api_key": api_key,
    }

    try:
        payload = await _serpapi_get(params)

        results = payload.get("news_results") or []
        if not results:
            return f"I couldn't find recent news results for {query}."

        summaries = []
        for item in results[:3]:
            title = item.get("title") or "Untitled"
            source = _source_name(item.get("source"))
            date = item.get("date")
            link = item.get("link")

            parts = [title]
            if source:
                parts.append(f"source: {source}")
            if date:
                parts.append(f"date: {date}")
            if link:
                parts.append(f"link: {link}")
            summaries.append("; ".join(parts))

        return "Latest news results: " + " | ".join(summaries)
    except TimeoutError:
        logger.warning("SerpApi news search timed out for query=%s", query)
        return "The live news search timed out. Please try again in a moment."
    except SerpApiError as exc:
        logger.warning(
            "SerpApi news search failed: status=%s message=%s query=%r location=%r",
            exc.status,
            exc.message,
            query,
            location,
        )
        return "The live news search failed, but your API key was not exposed in logs."
    except aiohttp.ClientError as exc:
        _log_serpapi_client_error(exc)
        return "The live news search failed because the search service could not be reached."
    except ValueError:
        logger.warning("SerpApi news search returned invalid JSON")
        return "The live news search returned an invalid response."


@function_tool
async def search_ai_mode(
    context: RunContext, query: str, location: str = "India"
) -> str:
    """Search Google AI Mode with SerpApi for broad web answers.

    Use this for non-news lookups, comparisons, explanations, India-specific
    questions, recommendations, and general web research where a synthesized
    answer with sources is helpful.
    """
    api_key = _clean_env("SERPAPI_API_KEY")
    if not api_key:
        return "I can't search Google AI Mode yet because SERPAPI_API_KEY is not configured."

    params = {
        "engine": "google_ai_mode",
        "q": query,
        "gl": _google_country_code(location),
        "hl": "en",
        "location": location,
        "api_key": api_key,
    }

    try:
        try:
            payload = await _serpapi_get(params, timeout_seconds=20)
        except SerpApiError as exc:
            if exc.status != 400 or location == "United States":
                raise

            logger.warning(
                "SerpApi AI Mode rejected location=%r for query=%r; retrying without location",
                location,
                query,
            )
            fallback_params = dict(params)
            fallback_params.pop("location", None)
            payload = await _serpapi_get(fallback_params, timeout_seconds=20)

        answer = payload.get("reconstructed_markdown")
        if not answer:
            snippets = [
                block.get("snippet", "")
                for block in payload.get("text_blocks", [])
                if block.get("snippet")
            ]
            answer = " ".join(snippets[:3])

        references = payload.get("references") or []
        source_parts = []
        for reference in references[:3]:
            title = reference.get("title") or "Untitled"
            source = reference.get("source")
            link = reference.get("link")
            parts = [title]
            if source:
                parts.append(f"source: {source}")
            if link:
                parts.append(f"link: {link}")
            source_parts.append("; ".join(parts))

        if not answer and not source_parts:
            return f"I couldn't find a Google AI Mode answer for {query}."

        response = f"Google AI Mode answer: {answer or 'No direct answer returned.'}"
        if source_parts:
            response += " Sources: " + " | ".join(source_parts)
        return response
    except TimeoutError:
        logger.warning("SerpApi AI Mode search timed out for query=%s", query)
        return "The Google AI Mode search timed out. Please try again in a moment."
    except SerpApiError as exc:
        logger.warning(
            "SerpApi AI Mode search failed: status=%s message=%s query=%r location=%r",
            exc.status,
            exc.message,
            query,
            location,
        )
        return "The Google AI Mode search failed, but your API key was not exposed in logs."
    except aiohttp.ClientError as exc:
        _log_serpapi_client_error(exc)
        return "The Google AI Mode search failed because the search service could not be reached."
    except ValueError:
        logger.warning("SerpApi AI Mode returned invalid JSON")
        return "The Google AI Mode search returned an invalid response."
