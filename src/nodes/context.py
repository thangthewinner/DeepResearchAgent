from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import CONTEXT_PRUNE_MAX_TOKENS
from ..logging_config import get_logger
from ..models.llm import compressor_model
from ..models.schemas import FactExtraction
from ..models.state import RAW_NOTES_CLEAR, SupervisorState
from ..prompts.context import CONTEXT_PRUNING_PROMPT

logger = get_logger(__name__)


async def context_pruning_node(state: SupervisorState) -> dict[str, Any]:
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
        result = await structured_llm.ainvoke(
            [HumanMessage(content=prompt)],
            config={"max_tokens": CONTEXT_PRUNE_MAX_TOKENS},
        )
        new_facts = result.new_facts

        message = (
            "[SYSTEM] Context Pruned. "
            f"{len(new_facts)} new facts added to Knowledge Base. Raw notes buffer cleared."
        )
    except Exception as e:
        logger.exception(
            "Context pruning failed",
            extra={"raw_notes_count": len(raw_notes), "error_type": type(e).__name__},
        )
        new_facts = []
        message = "[SYSTEM] Context Pruning failed."

    return {
        "raw_notes": [RAW_NOTES_CLEAR],
        "knowledge_base": new_facts,
        "supervisor_messages": [SystemMessage(content=message)],
    }
