import asyncio
import sys
import uuid
from pathlib import Path
from typing import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[1]))

import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from config.settings import REQUEST_TIMEOUT_SECONDS
from src.logging_config import get_logger
from src.runtime import bootstrap_agent


class SessionContext(TypedDict):
    """Per-browser session context for thread management."""

    session_id: str
    reset_counter: int


agent = bootstrap_agent()
logger = get_logger(__name__)


def _new_session_context() -> SessionContext:
    """Create a new session context."""
    return {"session_id": uuid.uuid4().hex, "reset_counter": 0}


def _session_thread_id(session_context: SessionContext) -> str:
    """Build a stable thread_id for the active session segment."""
    return f"gradio_{session_context['session_id']}_r{session_context['reset_counter']}"


async def _invoke_agent(
    session_context: SessionContext,
    user_text: str,
) -> tuple[str, SessionContext]:
    """Invoke agent for one turn with clarification-aware resume semantics."""
    active_session = dict(session_context)
    thread_id = _session_thread_id(active_session)
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(config)

    if state.next:
        logger.info("Resuming interrupted graph", extra={"thread_id": thread_id})
        coro = agent.ainvoke(Command(resume=user_text), config=config)
    elif state.values:
        old_thread_id = thread_id
        active_session["reset_counter"] += 1
        thread_id = _session_thread_id(active_session)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(
            "Previous session complete, rotating thread",
            extra={"old_thread_id": old_thread_id, "new_thread_id": thread_id},
        )
        coro = agent.ainvoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )
    else:
        logger.info("Starting new session", extra={"thread_id": thread_id})
        coro = agent.ainvoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

    result = await asyncio.wait_for(coro, timeout=REQUEST_TIMEOUT_SECONDS)

    new_state = await agent.aget_state(config)
    if new_state.next:
        question = new_state.tasks[0].interrupts[0].value
        logger.info("Agent needs clarification", extra={"thread_id": thread_id})
        return f"{question}", active_session

    report = result.get("final_report")
    if report:
        logger.info(
            "Research complete",
            extra={"thread_id": thread_id, "length": len(report)},
        )
        return str(report), active_session

    messages = result.get("messages", [])
    if messages:
        return str(messages[-1].content), active_session

    return "No result returned.", active_session


def build_app() -> gr.Blocks:
    """Build and return the Gradio app."""
    with gr.Blocks(title="DeepResearchAgent") as app:
        gr.Markdown(
            "# DeepResearchAgent\n"
            "Submit a research question and receive a structured Markdown report."
        )

        session_state = gr.State(value=_new_session_context())
        chatbot = gr.Chatbot(label="Research Chat", height=600)
        textbox = gr.Textbox(
            label="Your question",
            placeholder="Example: Compare leading AI models released in 2026.",
            lines=1,
        )
        send_btn = gr.Button("Send", variant="primary")

        async def _submit(
            user_text: str,
            history: list[dict[str, str]],
            session_context: SessionContext,
        ):
            message = user_text.strip()
            if not message:
                yield "", history, session_context
                return

            logger.info(
                "Received Gradio message",
                extra={
                    "thread_id": _session_thread_id(session_context),
                    "length": len(message),
                },
            )

            in_progress_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Thinking..."},
            ]
            yield "", in_progress_history, session_context

            try:
                response, updated_context = await _invoke_agent(
                    session_context, message
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Request timed out",
                    extra={"thread_id": _session_thread_id(session_context)},
                )
                response = (
                    "Request timed out "
                    f"({REQUEST_TIMEOUT_SECONDS}s). Please retry with a simpler question."
                )
                updated_context = session_context
            except Exception:
                logger.exception(
                    "Error handling Gradio message",
                    extra={"thread_id": _session_thread_id(session_context)},
                )
                response = "An error occurred. Please try again."
                updated_context = session_context

            final_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            yield "", final_history, updated_context

        send_btn.click(
            _submit,
            inputs=[textbox, chatbot, session_state],
            outputs=[textbox, chatbot, session_state],
        )
        textbox.submit(
            _submit,
            inputs=[textbox, chatbot, session_state],
            outputs=[textbox, chatbot, session_state],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
