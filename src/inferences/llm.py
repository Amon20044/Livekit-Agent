import os
from typing import Any

from livekit.agents import llm
from livekit.plugins import aws, google, groq

from core.env import _env_bool, _env_float, _env_int, _plugin_model
from settings import (
    gemini_fallback_model,
    gemini_thinking_level,
    google_api_key,
    groq_api_key,
)


def _build_llm_kwargs(model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": google_api_key,
        "vertexai": False,
        "temperature": _env_float(
            "GEMINI_TEMPERATURE", 0.35, min_value=0.0, max_value=2.0
        ),
        "max_output_tokens": _env_int("GEMINI_MAX_OUTPUT_TOKENS", 220, min_value=64),
    }

    if model.startswith("gemini-2.5"):
        kwargs["thinking_config"] = {
            "thinking_budget": _env_int("GEMINI_THINKING_BUDGET", 0, min_value=-1)
        }
    elif model.startswith("gemini-3"):
        kwargs["thinking_config"] = {"thinking_level": gemini_thinking_level}

    return kwargs


def _llm_provider() -> str:
    match os.getenv("LLM_PROVIDER", "google").strip().lower():
        case "aws" | "bedrock":
            return "bedrock"
        case "groq":
            return "groq"
        case _:
            return "google"


def _build_google_llm(model: str) -> llm.LLM:
    primary = google.LLM(**_build_llm_kwargs(model))

    if not _env_bool("GEMINI_FALLBACK_ENABLED", True):
        return primary

    fallback_model = _plugin_model(gemini_fallback_model, "google")
    if fallback_model == model:
        return primary

    fallback = google.LLM(**_build_llm_kwargs(fallback_model))
    return llm.FallbackAdapter(
        [primary, fallback],
        attempt_timeout=_env_float(
            "GEMINI_FALLBACK_ATTEMPT_TIMEOUT", 12.0, min_value=1.0, max_value=60.0
        ),
        max_retry_per_llm=_env_int(
            "GEMINI_FALLBACK_MAX_RETRY_PER_LLM", 0, min_value=0, max_value=3
        ),
        retry_interval=_env_float(
            "GEMINI_FALLBACK_RETRY_INTERVAL", 0.2, min_value=0.0, max_value=5.0
        ),
    )


class _LatencyOptimizedBedrockLLM(aws.LLM):
    """``aws.LLM`` that requests Bedrock latency-optimized inference.

    The plugin (v1.5) exposes no hook for the top-level Converse
    ``performanceConfig`` field, so we inject it into the stream options after
    ``chat()`` builds them but before the request is sent. Mutation is safe
    because nothing awaits between ``chat()`` returning and this assignment, so
    the stream's ``_run`` coroutine cannot have read the options yet.
    """

    def chat(self, **kwargs: Any) -> llm.LLMStream:
        stream = super().chat(**kwargs)
        stream._opts["performanceConfig"] = {"latency": "optimized"}
        return stream


def _build_bedrock_llm(model: str) -> aws.LLM:
    kwargs: dict[str, Any] = {
        "model": model,
        "region": os.getenv("AWS_BEDROCK_REGION", "us-east-1"),
        "temperature": _env_float(
            "BEDROCK_TEMPERATURE", 0.35, min_value=0.0, max_value=1.0
        ),
        "max_output_tokens": _env_int("BEDROCK_MAX_OUTPUT_TOKENS", 220, min_value=64),
    }
    if _env_bool("BEDROCK_LATENCY_OPTIMIZED", True):
        return _LatencyOptimizedBedrockLLM(**kwargs)
    return aws.LLM(**kwargs)


def _build_groq_llm(model: str) -> groq.LLM:
    return groq.LLM(
        model=model,
        api_key=groq_api_key,
        temperature=_env_float("GROQ_TEMPERATURE", 0.35, min_value=0.0, max_value=2.0),
        max_completion_tokens=_env_int("GROQ_MAX_OUTPUT_TOKENS", 220, min_value=64),
    )


def _build_llm(model: str) -> llm.LLM:
    match _llm_provider():
        case "bedrock":
            return _build_bedrock_llm(model)
        case "groq":
            return _build_groq_llm(model)
        case _:
            return _build_google_llm(model)
