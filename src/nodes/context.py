from langchain_core.messages import HumanMessage, SystemMessage

from ..models.llm import compressor_model
from ..models.schemas import FactExtraction
from ..models.state import SupervisorState
from ..prompts.context import CONTEXT_PRUNING_PROMPT


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
