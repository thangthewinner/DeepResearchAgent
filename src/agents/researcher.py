from typing import Any, Literal

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)

from config.settings import COMPRESS_MAX_TOKENS, RESEARCHER_LLM_MAX_TOKENS
from ..models.llm import base_model, compress_model

from ..models.state import ResearcherState
from ..prompts.researcher import (
    COMPRESS_HUMAN_MESSAGE_PROMPT,
    COMPRESS_RESEARCH_PROMPT,
    RESEARCH_AGENT_PROMPT,
)
from ..tools.registry import get_researcher_tools
from ..utils.date import get_today_str


def _get_runtime_tools() -> tuple[list[Any], dict[str, Any]]:
    """Return the current researcher tools and name lookup map."""
    tools = get_researcher_tools()
    return tools, {tool.name: tool for tool in tools}


def llm_call(state: ResearcherState):
    """
    Brain of researcher: analyzes state to branch off to an action/tool or finish
    """
    researcher_tools, _ = _get_runtime_tools()
    model_with_tools = base_model.bind_tools(researcher_tools)

    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=RESEARCH_AGENT_PROMPT.format(date=get_today_str())
                    )
                ]
                + list(state.get("researcher_messages", [])),
                config={"max_tokens": RESEARCHER_LLM_MAX_TOKENS},
            )
        ]
    }


def tool_node(state: ResearcherState):
    """
    Hands of researcher: executes tool calls
    """
    _, tools_by_name = _get_runtime_tools()
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool:
            observations.append(tool.invoke(tool_call["args"]))
        else:
            observations.append(f"Tool {tool_call['name']} not found")

    tool_outputs = [
        ToolMessage(
            content=str(observation),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def should_continue(
    state: ResearcherState,
) -> Literal["tool_node", "compress_research"]:
    """
    Continue ReAct loop or finish local research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return "compress_research"


def compress_research(state: ResearcherState) -> dict:
    """
    Final sub-graph node: compress loop findings into clean summary.
    """
    system_message = COMPRESS_RESEARCH_PROMPT.format(date=get_today_str())

    messages = (
        [SystemMessage(content=system_message)]
        + list(state.get("researcher_messages", []))
        + [
            HumanMessage(
                content=COMPRESS_HUMAN_MESSAGE_PROMPT.format(
                    research_topic=state.get("research_topic", "")
                )
            )
        ]
    )

    response = compress_model.invoke(
        messages, config={"max_tokens": COMPRESS_MAX_TOKENS}
    )

    raw_notes = [
        str(m.content)
        for m in filter_messages(
            state.get("researcher_messages", []), include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
    }
