"""
Centralized logging configuration for DeepResearchAgent.

Call setup_logging() once at application entry point (telegram_bot.py, simple_query.py).
All modules obtain a logger via: logger = logging.getLogger("deepresearch.<module>")

Environment variables:
  LOG_LEVEL  — INFO (default), DEBUG, WARNING, ERROR
  LOG_FILE   — Optional path to write rotating log file (e.g. logs/app.log).
               If not set, logs are only printed to stdout.
"""

import logging
import logging.config
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FILE = os.getenv("LOG_FILE", "").strip()

_handlers = ["console"]
_handler_cfg: dict = {
    "console": {
        "class": "logging.StreamHandler",
        "formatter": "standard",
        "stream": "ext://sys.stdout",
    },
}

if _LOG_FILE:
    import os as _os
    _os.makedirs(_os.path.dirname(_LOG_FILE) or ".", exist_ok=True)
    _handler_cfg["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "formatter": "standard",
        "filename": _LOG_FILE,
        "maxBytes": 10 * 1024 * 1024,  # 10 MB per file
        "backupCount": 5,
        "encoding": "utf-8",
    }
    _handlers.append("file")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": _handler_cfg,
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        # Application loggers — verbose
        "deepresearch": {
            "handlers": _handlers,
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
