"""
OpenWebUI-compatible FastAPI server for DeepResearchAgent.

Exposes an OpenAI-compatible Chat Completions API so OpenWebUI (or any
OpenAI-compatible client) can talk to the LangGraph agent.

Endpoints:
  GET  /v1/models                — model list required by OpenWebUI
  POST /v1/chat/completions      — main chat endpoint
  GET  /health                   — liveness probe

Run:
  uvicorn examples.webui_server:app --host 0.0.0.0 --port 8080 --reload

Thread management:
  OpenWebUI sends a unique chat ID in the header or we derive one from the
  conversation.  We map that to a LangGraph thread_id so the checkpointer
  preserves state across the clarification interrupt() loop.
"""

import asyncio
import os
import sys
import time
import uuid
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from config.settings import REQUEST_TIMEOUT_SECONDS  # noqa: E402
from src.graph import build_main_agent  # noqa: E402
from src.logging_config import get_logger, setup_logging  # noqa: E402

setup_logging()
logger = get_logger(__name__)

# Build the agent once at startup (reuses for all requests)
agent = build_main_agent()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DeepResearchAgent — OpenWebUI Server",
    description="OpenAI-compatible API wrapper for the LangGraph DeepResearchAgent.",
    version="1.0.0",
)

# CORS is required so the browser (OpenWebUI frontend) can call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your OpenWebUI URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# OpenAI-compatible schemas (minimal subset required by OpenWebUI)
# ---------------------------------------------------------------------------

MODEL_ID = "deep-research-agent"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    # Accept extra OpenAI params but ignore them
    model_config = {"extra": "allow"}


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = MODEL_ID
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "deepresearch"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_thread_id(request: Request) -> str:
    """
    Derive a stable thread_id from the request.

    OpenWebUI passes the chat session ID in the X-Session-ID header when
    configured to do so. We fall back to a random UUID (single-turn behaviour)
    if the header is absent.
    """
    session_id = (
        request.headers.get("X-Session-ID")
        or request.headers.get("x-session-id")
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    return f"webui_{session_id}"


async def _invoke_agent(thread_id: str, user_text: str) -> str:
    """
    Run one turn of the agent and return the text to send back to the user.

    Handles three cases:
      1. Graph is paused at interrupt() → resume with user's answer, then
         check if it paused again (more clarification) or produced a report.
      2. Previous session finished → rotate to a new thread so we start clean.
      3. Fresh session → standard ainvoke.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(config)

    if state.next:
        # Agent is waiting for clarification — resume with user text
        logger.info("Resuming interrupted graph", extra={"thread_id": thread_id})
        coro = agent.ainvoke(Command(resume=user_text), config=config)

    elif state.values:
        # Previous run completed — start fresh in a new thread
        new_thread_id = f"{thread_id}_r{uuid.uuid4().hex[:4]}"
        logger.info(
            "Previous session complete, rotating thread",
            extra={"old": thread_id, "new": new_thread_id},
        )
        config = {"configurable": {"thread_id": new_thread_id}}
        coro = agent.ainvoke(
            {"messages": [HumanMessage(content=user_text)]}, config=config
        )

    else:
        # Fresh session
        logger.info("Starting new session", extra={"thread_id": thread_id})
        coro = agent.ainvoke(
            {"messages": [HumanMessage(content=user_text)]}, config=config
        )

    result = await asyncio.wait_for(coro, timeout=REQUEST_TIMEOUT_SECONDS)

    # Check if the graph paused again (clarification question)
    new_state = await agent.aget_state(config)
    if new_state.next:
        question = new_state.tasks[0].interrupts[0].value
        logger.info("Agent needs clarification", extra={"question": question})
        return f"❓ {question}"

    # Final report
    if "final_report" in result and result["final_report"]:
        logger.info(
            "Research complete",
            extra={"thread_id": thread_id, "length": len(result["final_report"])},
        )
        return result["final_report"]

    # Fallback: last message
    messages = result.get("messages", [])
    return messages[-1].content if messages else "⚠️ Không có kết quả."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Required by OpenWebUI to populate the model selector."""
    return ModelList(data=[ModelInfo(id=MODEL_ID)])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    OpenWebUI sends the full message history on every turn; we take only
    the last user message as the new input and let LangGraph's checkpointer
    manage the full conversation state internally.
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    # Extract the latest user message
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    user_text = user_messages[-1].content
    thread_id = _get_thread_id(request)

    logger.info(
        "Chat completion request",
        extra={"thread_id": thread_id, "length": len(user_text)},
    )

    try:
        answer = await _invoke_agent(thread_id, user_text)
    except asyncio.TimeoutError:
        logger.error("Request timed out", extra={"thread_id": thread_id})
        raise HTTPException(
            status_code=504,
            detail=f"Agent timed out after {REQUEST_TIMEOUT_SECONDS}s. "
            "Try a simpler or more specific question.",
        )
    except Exception as exc:
        logger.exception("Agent error", extra={"thread_id": thread_id})
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatCompletionResponse(
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=answer)
            )
        ]
    )
