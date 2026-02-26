from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool

from config.settings import OPENAI_API_KEY, WRITER_MODEL

from ..models.state import AgentState
from ..prompts.writer import (
    FINAL_REPORT_GENERATION_WITH_HELPFULNESS_INSIGHTFULNESS_HIT_CITATION_PROMPT,
    REPORT_GENERATION_WITH_DRAFT_INSIGHT_PROMPT,
)
from ..utils.date import get_today_str

writer_model = init_chat_model(model=WRITER_MODEL, api_key=OPENAI_API_KEY)


@tool(parse_docstring=True)
def refine_draft_report(
    research_brief: Annotated[str, InjectedToolArg],
    findings: Annotated[str, InjectedToolArg],
    draft_report: Annotated[str, InjectedToolArg],
):
    """Refine draft report

    Synthesizes all research findings into a comprehensive draft report

    Args:
        research_brief: user's research request
        findings: collected research findings for the user request
        draft_report: draft report based on the findings and user request

    Returns:
        refined draft report
    """
    draft_report_prompt = REPORT_GENERATION_WITH_DRAFT_INSIGHT_PROMPT.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str(),
    )

    draft_report_response = writer_model.invoke(
        [HumanMessage(content=draft_report_prompt)]
    )
    return draft_report_response.content


def final_report_generation(state: AgentState) -> dict:
    """
    Final node of the main graph. Synthesizes all gathered information into final report.
    """
    research_brief = state.get("research_brief", "")
    findings = "\n".join(state.get("notes", []))
    draft_report = state.get("draft_report", "")

    prompt = FINAL_REPORT_GENERATION_WITH_HELPFULNESS_INSIGHTFULNESS_HIT_CITATION_PROMPT.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str(),
    )

    response = writer_model.invoke([HumanMessage(content=prompt)])

    return {"final_report": response.content}
