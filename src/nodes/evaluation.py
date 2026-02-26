from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import CRITIC_MODEL, JUDGE_MODEL, OPENAI_API_KEY

from ..models.schemas import Critique, EvaluationResult
from ..models.state import SupervisorState
from ..prompts.evaluation import EVALUATE_DRAFT_PROMPT, RED_TEAM_PROMPT

# We define a specialized, powerful model for our critic and judge.
critic_model = init_chat_model(model=CRITIC_MODEL, api_key=OPENAI_API_KEY)
judge_model = init_chat_model(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)


async def red_team_node(state: SupervisorState) -> dict:
    """
    'Red Team' agent node. Critiques current draft to find logical flaws / biases.
    """
    draft = state.get("draft_report", "")

    if not draft or len(draft) < 50:
        return {}

    prompt = RED_TEAM_PROMPT.format(draft=draft)

    response = await critic_model.ainvoke([HumanMessage(content=prompt)])
    content = response.content

    if "PASS" in content and len(content) < 20:
        return {}

    critique = Critique(
        author="Red Team Adversary", concern=content, severity=8, addressed=False
    )

    return {
        "active_critiques": [critique],
        "supervisor_messages": [
            SystemMessage(content=f"⚠️ ADVERSARIAL FEEDBACK DETECTED: {content}")
        ],
    }


def evaluate_draft_quality(research_brief: str, draft_report: str) -> EvaluationResult:
    """
    'Self-Evolution' scoring mechanism. LLM-as-a-judge quality evaluate.
    """

    eval_prompt = EVALUATE_DRAFT_PROMPT.format(
        research_brief=research_brief, draft_report=draft_report
    )

    structured_judge = judge_model.with_structured_output(EvaluationResult)

    return structured_judge.invoke([HumanMessage(content=eval_prompt)])
