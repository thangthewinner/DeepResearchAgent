"""
Centralized logging configuration for DeepResearchAgent.

Call setup_logging() once at application entry point (telegram_bot.py, simple_query.py).
All modules obtain a logger via: logger = logging.getLogger("deepresearch.<module>")
"""

import logging
import logging.config
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        # Application loggers — verbose
        "deepresearch": {
            "handlers": ["console"],
            "level": _LOG_LEVEL,
            "propagate": False,
        },
        # Silence noisy third-party libs
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "openai": {"level": "WARNING"},
        "langchain": {"level": "WARNING"},
        "langgraph": {"level": "WARNING"},
        "telegram": {"level": "INFO"},
    },
}


def setup_logging() -> None:
    """Apply logging configuration. Call once at startup."""
    logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """
    Return a namespaced logger.

    Usage:
        from src.logging_config import get_logger
        logger = get_logger(__name__)
    """
    # Ensure module name is prefixed with "deepresearch" for unified filtering
    if not name.startswith("deepresearch"):
        # Convert src.tools.search → deepresearch.tools.search
        name = "deepresearch." + name.replace("src.", "")
    return logging.getLogger(name)
