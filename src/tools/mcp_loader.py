import shlex
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from config.settings import (
    MCP_ALLOWED_TOOLS,
    MCP_ENABLED,
    MCP_SERVER_ARGS,
    MCP_SERVER_COMMAND,
    MCP_SERVER_TRANSPORT,
    MCP_SERVER_URL,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args(value: str) -> list[str]:
    try:
        return shlex.split(value)
    except ValueError:
        logger.exception("Invalid MCP server args")
        return []


async def load_mcp_tools() -> list[Any]:
    """Load LangChain-compatible tools from MCP servers."""
    if not MCP_ENABLED:
        return []

    if MCP_SERVER_TRANSPORT == "stdio":
        if not MCP_SERVER_COMMAND:
            logger.warning("MCP enabled but command is empty")
            return []
        server_config = {
            "deepresearch": {
                "transport": "stdio",
                "command": MCP_SERVER_COMMAND,
                "args": _parse_args(MCP_SERVER_ARGS),
            }
        }
    elif MCP_SERVER_TRANSPORT == "streamable_http":
        if not MCP_SERVER_URL:
            logger.warning("MCP enabled but URL is empty")
            return []
        server_config = {
            "deepresearch": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
            }
        }
    else:
        logger.warning(
            "Unknown MCP transport",
            extra={"transport": MCP_SERVER_TRANSPORT},
        )
        return []

    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()

    allowlist = set(_parse_csv(MCP_ALLOWED_TOOLS))
    if allowlist:
        tools = [tool for tool in tools if tool.name in allowlist]

    logger.info("Loaded MCP tools", extra={"count": len(tools)})
    return tools
