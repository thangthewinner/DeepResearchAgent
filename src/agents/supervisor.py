import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.graph import END
from langgraph.types import Command

from config.settings import BASE_MODEL, OPENAI_API_KEY

from ..models.schemas import QualityMetric
from ..models.state import SupervisorState
from ..nodes.evaluation import evaluate_draft_quality
from ..prompts.supervisor import (
    LEAD_RESEARCHER_WITH_MULTIPLE_STEPS_DIFFUSION_DOUBLE_CHECK_PROMPT,
)
from ..tools.common import ConductResearch, ResearchComplete, think_tool
from ..utils.date import get_today_str

# Initialize supervisor tools
from .writer import refine_draft_report

supervisor_tools = [
    ConductResearch,
    ResearchComplete,
    think_tool,
    refine_draft_report,
]

base_model = init_chat_model(model=BASE_MODEL, api_key=OPENAI_API_KEY)
supervisor_model_with_tools = base_model.bind_tools(supervisor_tools)

max_concurrent_researchers = 3
max_researcher_iterations = 5

# Will be injected from graph.py builder to avoid circular dependency
researcher_agent = None


def get_notes_from_tool_calls(messages: list) -> list[str]:
    """Helper extract tool string content log from tool messages."""
    return [
        tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")
    ]


async def supervisor_node(
    state: SupervisorState,
) -> Command[Literal["supervisor_tools"]]:
    """
    Brain of Diffusion. Analyzes state/critical feedback and plans tools.
    """
    supervisor_messages = list(state.get("supervisor_messages", []))

    system_message = (
        LEAD_RESEARCHER_WITH_MULTIPLE_STEPS_DIFFUSION_DOUBLE_CHECK_PROMPT.format(
            date=get_today_str(),
            max_concurrent_research_units=max_concurrent_researchers,
            max_researcher_iterations=max_researcher_iterations,
        )
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # Active Critiques Self-Correction
    critiques = state.get("active_critiques", [])
    unaddressed = [c for c in critiques if not c.addressed]
    if unaddressed:
        critique_text = "\n".join(
            [f"- {c.author} says: {c.concern}" for c in unaddressed]
        )
        intervention = SystemMessage(
            content=f"""
        CRITICAL INTERVENTION REQUIRED.
        The following issues were detected by the Adversarial Team in your draft:
        {critique_text}
        
        You MUST address these issues in your next step.
        If the critique says citations are missing, call 'ConductResearch'.
        If the critique says logic is flawed, call 'think_tool'.
        """
        )
        messages.append(intervention)

    if state.get("needs_quality_repair"):
        messages.append(
            SystemMessage(
                content="PREVIOUS DRAFT QUALITY WAS LOW (Score < 7/10). Focus on finding new sources and citing them."
            )
        )

    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            "needs_quality_repair": False,
        },
    )


async def supervisor_tools_node(
    state: SupervisorState,
) -> Command[Literal["red_team", "context_pruner", "__end__"]]:
    """
    Hands of Supervisor. Execute fan-out tool calls.
    """
    most_recent_message = state.get("supervisor_messages", [])[-1]

    exceeded_iterations = (
        state.get("research_iterations", 0) >= max_researcher_iterations
    )
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        kb_notes = [
            f"{f.content} (Confidence: {f.confidence_score})"
            for f in state.get("knowledge_base", [])
        ]
        if not kb_notes:
            kb_notes = get_notes_from_tool_calls(state.get("supervisor_messages", []))

        return Command(
            goto=END,
            update={
                "notes": kb_notes,
                "research_brief": state.get("research_brief", ""),
            },
        )

    conduct_research_calls = [
        t for t in most_recent_message.tool_calls if t["name"] == "ConductResearch"
    ]
    refine_report_calls = [
        t for t in most_recent_message.tool_calls if t["name"] == "refine_draft_report"
    ]
    think_calls = [
        t for t in most_recent_message.tool_calls if t["name"] == "think_tool"
    ]

    tool_messages = []
    all_raw_notes = []
    draft_report = state.get("draft_report", "")
    updates = {}

    for tool_call in think_calls:
        observation = think_tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(
                content=observation, name="think_tool", tool_call_id=tool_call["id"]
            )
        )

    if conduct_research_calls and researcher_agent is not None:
        coros = [
            researcher_agent.ainvoke(
                {
                    "researcher_messages": [
                        HumanMessage(content=tc["args"]["research_topic"])
                    ],
                    "research_topic": tc["args"]["research_topic"],
                }
            )
            for tc in conduct_research_calls
        ]
        results = await asyncio.gather(*coros)
        for result, tool_call in zip(results, conduct_research_calls):
            tool_messages.append(
                ToolMessage(
                    content=result.get("compressed_research", ""),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            all_raw_notes.extend(result.get("raw_notes", []))

    for tool_call in refine_report_calls:
        kb = state.get("knowledge_base", [])
        kb_str = (
            "CONFIRMED FACTS:\n" + "\n".join([f"- {f.content}" for f in kb])
            if kb
            else "\n".join(
                get_notes_from_tool_calls(state.get("supervisor_messages", []))
            )
        )
        new_draft = refine_draft_report.invoke(
            {
                "research_brief": state.get("research_brief", ""),
                "findings": kb_str,
                "draft_report": state.get("draft_report", ""),
            }
        )

        eval_result = evaluate_draft_quality(
            research_brief=state.get("research_brief", ""), draft_report=new_draft
        )
        avg_score = (
            eval_result.comprehensiveness_score + eval_result.accuracy_score
        ) / 2

        tool_messages.append(
            ToolMessage(
                content=f"Draft Updated.\nQuality Score: {avg_score}/10.\nJudge Feedback: {eval_result.specific_critique}",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
        draft_report = new_draft

        updates["quality_history"] = [
            QualityMetric(
                score=avg_score,
                feedback=eval_result.specific_critique,
                iteration=state.get("research_iterations", 0),
            )
        ]
        if avg_score < 7.0:
            updates["needs_quality_repair"] = True

    updates["supervisor_messages"] = tool_messages
    updates["raw_notes"] = all_raw_notes
    updates["draft_report"] = draft_report

    return Command(goto=["red_team", "context_pruner"], update=updates)
