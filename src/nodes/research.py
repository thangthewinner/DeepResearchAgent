from langchain_core.messages import HumanMessage

from ..models.llm import creative_model
from ..models.schemas import DraftReport
from ..models.state import AgentState
from ..prompts.writer import DRAFT_REPORT_GENERATION_PROMPT
from ..utils.date import get_today_str


def write_draft_report(state: AgentState) -> dict:
    """
    Takes research brief and generates initial, unresearched draft.
    """
    structured_output_model = creative_model.with_structured_output(DraftReport)
    research_brief = state.get("research_brief", "")

    draft_report_prompt_formatted = DRAFT_REPORT_GENERATION_PROMPT.format(
        research_brief=research_brief, date=get_today_str()
    )

    response = structured_output_model.invoke(
        [HumanMessage(content=draft_report_prompt_formatted)]
    )

    return {
        "research_brief": research_brief,
        "draft_report": response.draft_report,
        "supervisor_messages": [
            HumanMessage(content="Here is the draft report: " + response.draft_report),
            HumanMessage(content=research_brief),
        ],
    }
