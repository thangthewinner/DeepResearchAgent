from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import COMPRESSOR_MODEL, OPENAI_API_KEY

from ..models.schemas import FactExtraction
from ..models.state import SupervisorState
from ..prompts.context import CONTEXT_PRUNING_PROMPT

# We use a fast, cheaper model for routine extraction
compressor_model = init_chat_model(model=COMPRESSOR_MODEL, api_key=OPENAI_API_KEY)


async def context_pruning_node(state: SupervisorState) -> dict:
    """
    'Context Engineering'. Takes raw notes buffer, extracts facts, clears buffer.
    """
    raw_notes = state.get("raw_notes", [])

    if not raw_notes:
        return {}

    text_block = "\n".join(raw_notes)
    prompt = CONTEXT_PRUNING_PROMPT.format(text_block=text_block[:20000])

    try:
        structured_llm = compressor_model.with_structured_output(FactExtraction)
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        new_facts = result.new_facts

        message = f"[SYSTEM] Context Pruned. {len(new_facts)} new facts added to Knowledge Base. Raw notes buffer cleared."
    except Exception as e:
        new_facts = []
        message = f"[SYSTEM] Context Pruning failed: {str(e)}"

    return {
        "raw_notes": [],
        "knowledge_base": new_facts,
        "supervisor_messages": [SystemMessage(content=message)],
    }
