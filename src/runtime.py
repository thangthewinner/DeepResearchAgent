import asyncio
import concurrent.futures
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .logging_config import get_logger, setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger(__name__)


def _run_async(coro):
    """Run an async coroutine in a sync context.

    Handles the case where an event loop may already be running
    (e.g., when called from an async framework like Gradio).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside an async context — run in a separate thread
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result()


def bootstrap_agent() -> Any:
    """Load environment, configure logging, and build the main agent graph."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    setup_logging()

    from .graph import build_main_agent

    return build_main_agent()
