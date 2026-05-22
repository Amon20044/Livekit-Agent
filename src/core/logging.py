import logging
import os


def _configure_log_levels() -> None:
    """Apply LOG_LEVEL to our and LiveKit's loggers to cut log volume.

    Lower verbosity (e.g. ``warn``) reduces per-turn logging work on the hot
    path. Levels are set on named loggers so the choice survives whatever the
    LiveKit CLI configures on the root logger.
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = levels.get(os.getenv("LOG_LEVEL", "info").strip().upper(), logging.INFO)
    for name in ("agent", "tools", "livekit", "livekit.agents"):
        logging.getLogger(name).setLevel(level)
