import asyncio
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .graph import build_main_agent
from .logging_config import get_logger, setup_logging
from .tools.mcp_loader import load_mcp_tools
from .tools.registry import set_external_research_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)
logger = get_logger(__name__)


def bootstrap_agent() -> Any:
    """Load environment, configure logging, and build the main agent graph."""
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    setup_logging()

    try:
        mcp_tools = asyncio.run(load_mcp_tools())
    except Exception:
        logger.exception("Failed to load MCP tools")
        mcp_tools = []

    set_external_research_tools(mcp_tools)
    logger.info("Registered external research tools", extra={"count": len(mcp_tools)})

    return build_main_agent()
