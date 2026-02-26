from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """
    Tool for strategic reflection on research progress.

    Use this after each search to analyze results and plan next steps.

    Args:
        reflection: Detailed reflection on progress, findings, gaps, and next steps.

    Returns:
        Confirmation that reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"


@tool
class ConductResearch(BaseModel):
    """Delegate research task to specialized sub-agent."""

    research_topic: str = Field(
        description="Single, self-contained topic described in high detail.",
    )


@tool
class ResearchComplete(BaseModel):
    """Signal that research process is complete."""

    pass
