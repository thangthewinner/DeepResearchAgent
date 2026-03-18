from typing import Any

from .common import think_tool
from .search import tavily_search

_external_research_tools: list[Any] = []


def set_external_research_tools(tools: list[Any]) -> None:
    global _external_research_tools
    _external_research_tools = tools


def get_researcher_tools() -> list[Any]:
    return [tavily_search, think_tool, *_external_research_tools]
