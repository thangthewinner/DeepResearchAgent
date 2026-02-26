from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import END
from langgraph.types import Command

from config.settings import BASE_MODEL, OPENAI_API_KEY

from ..models.schemas import ClarifyWithUser, ResearchQuestion
from ..models.state import AgentState
from ..prompts.clarification import (
    CLARIFY_WITH_USER_INSTRUCTIONS,
    TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_HUMAN_MSG_PROMPT,
)
from ..utils.date import get_today_str

model = init_chat_model(model=BASE_MODEL, api_key=OPENAI_API_KEY)


def clarify_with_user(
    state: AgentState,
) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Gatekeeper node. Determines if the user's request has enough detail to proceed.
    """
    messages_text = get_buffer_string(state.get("messages", []))
    current_date = get_today_str()

    structured_output_model = model.with_structured_output(ClarifyWithUser)

    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=CLARIFY_WITH_USER_INSTRUCTIONS.format(
                    messages=messages_text, date=current_date
                )
            )
        ]
    )

    if response.need_clarification:
        return Command(
            goto=END, update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """
    Transforms confirmed conversation history into a single research brief.
    """
    structured_output_model = model.with_structured_output(ResearchQuestion)

    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_HUMAN_MSG_PROMPT.format(
                    messages=get_buffer_string(state.get("messages", [])),
                    date=get_today_str(),
                )
            )
        ]
    )

    return Command(
        goto="write_draft_report", update={"research_brief": response.research_brief}
    )
