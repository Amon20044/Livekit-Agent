from typing import Any

from core.env import _env_float


def _money(value: float) -> str:
    return f"${value:.6f}"


def _pricing_config() -> dict[str, float]:
    return {
        "stt_per_minute": _env_float(
            "COST_DEEPGRAM_STT_PER_MINUTE_USD",
            0.0,
            min_value=0.0,
        ),
        "gemini_input_per_1m": _env_float(
            "COST_GEMINI_INPUT_PER_1M_TOKENS_USD",
            0.10,
            min_value=0.0,
        ),
        "gemini_output_per_1m": _env_float(
            "COST_GEMINI_OUTPUT_PER_1M_TOKENS_USD",
            0.40,
            min_value=0.0,
        ),
        "sarvam_tts_per_1k_chars": _env_float(
            "COST_SARVAM_TTS_PER_1K_CHARS_USD",
            0.02,
            min_value=0.0,
        ),
        "elevenlabs_tts_per_1k_chars": _env_float(
            "COST_ELEVENLABS_TTS_PER_1K_CHARS_USD",
            0.05,
            min_value=0.0,
        ),
    }


def _usage_costs(
    usage: Any,
    pricing: dict[str, float],
    *,
    tts_provider: str = "sarvam",
) -> dict[str, float]:
    costs = {"stt": 0.0, "llm": 0.0, tts_provider: 0.0}

    if usage.type == "stt_usage":
        costs["stt"] = usage.audio_duration / 60.0 * pricing["stt_per_minute"]
    elif usage.type == "llm_usage":
        billable_input_tokens = max(usage.input_tokens - usage.input_cached_tokens, 0)
        costs["llm"] = (
            billable_input_tokens / 1_000_000.0 * pricing["gemini_input_per_1m"]
            + usage.output_tokens / 1_000_000.0 * pricing["gemini_output_per_1m"]
        )
    elif usage.type == "tts_usage":
        costs[tts_provider] = (
            usage.characters_count
            / 1_000.0
            * pricing[f"{tts_provider}_tts_per_1k_chars"]
        )

    return costs


def _session_costs(
    session_usage: Any,
    pricing: dict[str, float],
    *,
    tts_provider: str = "sarvam",
) -> dict[str, float]:
    totals = {"stt": 0.0, "llm": 0.0, tts_provider: 0.0}

    for usage in session_usage.model_usage:
        for provider, cost in _usage_costs(
            usage, pricing, tts_provider=tts_provider
        ).items():
            totals[provider] += cost

    totals["total"] = sum(totals.values())
    return totals


def _cost_delta(
    current: dict[str, float],
    previous: dict[str, float],
) -> dict[str, float]:
    keys = [
        key
        for key in ("stt", "llm", "sarvam", "elevenlabs", "total")
        if key in current or key in previous
    ]
    return {
        key: max(current.get(key, 0.0) - previous.get(key, 0.0), 0.0) for key in keys
    }


def _format_cost_summary(costs: dict[str, float]) -> str:
    parts = [
        f"stt={_money(costs['stt'])}",
        f"llm={_money(costs['llm'])}",
    ]
    for provider in ("sarvam", "elevenlabs"):
        if provider in costs:
            parts.append(f"{provider}={_money(costs[provider])}")
    parts.append(f"total={_money(costs['total'])}")
    return " ".join(parts)


def _loggable_costs(costs: dict[str, float]) -> dict[str, str]:
    return {
        key: _money(costs[key])
        for key in ("stt", "llm", "sarvam", "elevenlabs", "total")
        if key in costs
    }
