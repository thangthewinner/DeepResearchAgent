from config.settings import OPENAI_API_KEY, SUMMARIZATION_MODEL
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ..models.schemas import Summary
from .date import get_today_str

summarization_model = init_chat_model(model=SUMMARIZATION_MODEL, api_key=OPENAI_API_KEY)

SUMMARIZE_WEBPAGE_PROMPT = """You are tasked with summarizing webpage content. Your goal is to preserve the most important information.

<webpage_content>
{webpage_content}
</webpage_content>

Create a summary that:
1. Preserves main topic and purpose
2. Retains key facts, statistics, data points
3. Keeps important quotes from credible sources
4. Maintains chronological order if time-sensitive
5. Includes relevant dates, names, locations

Aim for 25-30% of original length unless already concise.

Today's date is {date}.
"""


def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using LLM."""
    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary_result = structured_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEBPAGE_PROMPT.format(
                        webpage_content=webpage_content, date=get_today_str()
                    )
                )
            ]
        )
        formatted_summary = (
            f"<summary>\n{summary_result.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary_result.key_excerpts}\n</key_excerpts>"
        )
        return formatted_summary
    except Exception as e:
        print(f"❌ Summarization failed: {str(e)}")
        return (
            webpage_content[:1000] + "..."
            if len(webpage_content) > 1000
            else webpage_content
        )
