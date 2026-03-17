from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from config.settings import BRIEF_MAX_TOKENS, CLARIFY_MAX_TOKENS
from ..models.llm import base_model as model

from ..models.schemas import ClarifyWithUser, ResearchQuestion
from ..models.state import AgentState
from ..prompts.clarification import (
    CLARIFY_WITH_USER_INSTRUCTIONS,
    TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_HUMAN_MSG_PROMPT,
)
from ..utils.date import get_today_str


def clarify_with_user(
    state: AgentState,
) -> Command[Literal["clarify_with_user", "write_research_brief"]]:
    """
    Gatekeeper node. Determines if the user's request has enough detail to proceed.
    Uses interrupt() to pause graph and wait for user input when clarification is needed.
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
        ],
        config=RunnableConfig(max_tokens=CLARIFY_MAX_TOKENS),
    )

    if response.need_clarification:
        # Pause graph, send question to caller (e.g. Telegram bot)
        # When user replies, interrupt() returns the user's answer
        user_answer = interrupt(response.question)

        return Command(
            goto="clarify_with_user",
            update={
                "messages": [
                    AIMessage(content=response.question),
                    HumanMessage(content=user_answer),
                ]
            },
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
        ],
        config={"max_tokens": BRIEF_MAX_TOKENS},
    )

    return Command(
        goto="write_draft_report", update={"research_brief": response.research_brief}
    )
