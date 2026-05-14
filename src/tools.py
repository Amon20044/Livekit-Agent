import logging
import os
from typing import Any

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


async def _announce_search(context: RunContext | None) -> None:
    if context is None:
        return
    if getattr(context, "_anchor_search_announced", False):
        return

    context._anchor_search_announced = True

    handle = context.session.say(
        "I'll check that now.",
        allow_interruptions=False,
        add_to_chat_ctx=False,
    )
    await handle.wait_for_playout()


async def _serpapi_get(params: dict[str, str], timeout_seconds: int = 12) -> dict:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.get(SERPAPI_URL, params=params) as response,
    ):
        response.raise_for_status()
        return await response.json()


def _source_name(source: Any) -> str | None:
    if isinstance(source, dict):
        return source.get("name")
    if isinstance(source, str):
        return source
    return None


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

    await _announce_search(context)

    params = {
        "engine": "google_news",
        "q": query,
        "gl": "us",
        "hl": "en",
        "location": location,
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
    except (aiohttp.ClientError, ValueError) as exc:
        logger.exception("SerpApi news search failed")
        return f"The live news search failed: {exc}"


@function_tool
async def search_ai_mode(
    context: RunContext, query: str, location: str = "United States"
) -> str:
    """Search Google AI Mode with SerpApi for broad web answers.

    Use this for non-news lookups, comparisons, explanations, recommendations,
    and general web research where a synthesized answer with sources is helpful.
    """
    api_key = _clean_env("SERPAPI_API_KEY")
    if not api_key:
        return "I can't search Google AI Mode yet because SERPAPI_API_KEY is not configured."

    await _announce_search(context)

    params = {
        "engine": "google_ai_mode",
        "q": query,
        "gl": "us",
        "hl": "en",
        "location": location,
        "api_key": api_key,
    }

    try:
        payload = await _serpapi_get(params, timeout_seconds=20)

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
    except (aiohttp.ClientError, ValueError) as exc:
        logger.exception("SerpApi AI Mode search failed")
        return f"The Google AI Mode search failed: {exc}"
