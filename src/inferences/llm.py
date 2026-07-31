from livekit.plugins import openai
from openai.types.shared_params.reasoning import Reasoning

from app.models import ModelConfig
from settings import OPENAI_API_KEY


def build_llm(config: ModelConfig, api_key: str = "") -> openai.responses.LLM:
    """Use OpenAI's Responses API; the agent node owns the selected model."""
    effort = config.reasoning_effort
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        effort = "none"
    return openai.responses.LLM(
        model=config.name,
        api_key=api_key or OPENAI_API_KEY,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        reasoning=Reasoning(effort=effort),
        store=False,
    )
