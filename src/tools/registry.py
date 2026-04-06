from __future__ import annotations

from typing import Any

from .common import think_tool
from .search import tavily_search


def get_researcher_tools() -> list[Any]:
    """Return native tools for the researcher agent."""
    return [tavily_search, think_tool]
