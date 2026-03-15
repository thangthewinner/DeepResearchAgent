import operator
from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

from .schemas import Critique, Fact, QualityMetric

RAW_NOTES_CLEAR = "__CLEAR_RAW_NOTES_BUFFER_INTERNAL__"


def merge_raw_notes(existing: list[str], update: list[str]) -> list[str]:
    """Append incoming notes or clear the raw notes buffer."""
    if update == [RAW_NOTES_CLEAR]:
        return []
    return existing + update


class ResearcherState(TypedDict):
    """Worker research agent state."""

    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], merge_raw_notes]


class ResearcherOutputState(TypedDict):
    """Research agent output."""

    compressed_research: str
    raw_notes: Annotated[List[str], merge_raw_notes]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]


class SupervisorState(TypedDict):
    """Main Supervisor state."""

    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    draft_report: str
    raw_notes: Annotated[List[str], merge_raw_notes]
    knowledge_base: Annotated[List[Fact], operator.add]
    research_iterations: int
    active_critiques: Annotated[List[Critique], operator.add]
    quality_history: Annotated[List[QualityMetric], operator.add]
    needs_quality_repair: bool


class AgentInputState(MessagesState):
    """Initial input state."""

    pass


class AgentState(MessagesState):
    """Main multi-agent state."""

    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], merge_raw_notes] = []
    notes: Annotated[list[str], operator.add] = []
    knowledge_base: Annotated[List[Fact], operator.add] = []
    draft_report: str
    final_report: str
