from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool

from config.settings import FINAL_REPORT_MAX_TOKENS, REFINE_MAX_TOKENS
from ..models.llm import writer_model

from ..models.state import AgentState
from ..prompts.writer import (
    FINAL_REPORT_GENERATION_WITH_HELPFULNESS_INSIGHTFULNESS_HIT_CITATION_PROMPT,
    REPORT_GENERATION_WITH_DRAFT_INSIGHT_PROMPT,
)
from ..utils.date import get_today_str


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
        [HumanMessage(content=draft_report_prompt)],
        config={"max_tokens": REFINE_MAX_TOKENS},
    )
    return draft_report_response.content


def final_report_generation(state: AgentState) -> dict:
    """
    Final node of the main graph. Synthesizes all gathered information into final report.
    """
    research_brief = state.get("research_brief", "")
    draft_report = state.get("draft_report", "")

    # Combine structured knowledge_base facts with raw notes
    kb_facts = state.get("knowledge_base", [])
    kb_section = "\n".join(
        [
            f"- [{f.confidence_score}%] {f.content} (source: {f.source_url})"
            for f in kb_facts
        ]
    )
    raw_notes = "\n".join(state.get("notes", []))

    findings = ""
    if kb_section:
        findings += "VERIFIED FACTS:\n" + kb_section + "\n\n"
    if raw_notes:
        findings += "ADDITIONAL NOTES:\n" + raw_notes

    prompt = FINAL_REPORT_GENERATION_WITH_HELPFULNESS_INSIGHTFULNESS_HIT_CITATION_PROMPT.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str(),
    )

    response = writer_model.invoke(
        [HumanMessage(content=prompt)],
        config={"max_tokens": FINAL_REPORT_MAX_TOKENS},
    )

    return {"final_report": response.content}
