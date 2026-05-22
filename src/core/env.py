import logging
import os

logger = logging.getLogger("agent")


def _plugin_model(value: str, provider_prefix: str) -> str:
    model = value.strip().strip('"').strip("'")
    if model.startswith(f"{provider_prefix}/"):
        model = model.split("/", 1)[1]
    if provider_prefix == "deepgram" and model == "nova-3-general":
        return "nova-3"
    return model


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = os.getenv(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value.strip())
        except ValueError:
            logger.warning("Invalid float for %s=%r; using %.3f", name, value, default)
            parsed = default

    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = os.getenv(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value.strip())
        except ValueError:
            logger.warning("Invalid int for %s=%r; using %d", name, value, default)
            parsed = default

    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed
