import asyncio
import os
import sys
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config.settings import REQUEST_TIMEOUT_SECONDS
from src.graph import build_main_agent
from src.logging_config import get_logger, setup_logging
from src.server import run_server

# LangChain's with_structured_output() stores parsed results in AIMessage.parsed
# which Pydantic declares as None, causing harmless serialization warnings.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

setup_logging()
logger = get_logger(__name__)

# MemorySaver: state lives in RAM only.
# Restart → clean slate for all sessions (intentional design choice).
agent = build_main_agent()

# Background task strong-reference set (prevents GC)
_background_tasks: set = set()


async def post_init(application: Application) -> None:
    """Start the FastAPI health check server as a background task."""
    task = asyncio.create_task(run_server())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    logger.info("Telegram bot starting...")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Xin chào! Gửi câu hỏi nghiên cứu cho tôi, tôi sẽ phân tích và trả lời chi tiết."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = str(update.effective_chat.id)
    reset_counter = context.user_data.get("reset_counter", 0)
    config = {"configurable": {"thread_id": f"tg_{chat_id}_r{reset_counter}"}}

    logger.info("Received message", extra={"chat_id": chat_id, "length": len(user_text)})
    await update.message.reply_text("🔍 Đang xử lý... Vui lòng chờ.")

    try:
        state = await agent.aget_state(config)

        if state.next:
            # Graph is paused at interrupt() — resume with user's answer
            coro = agent.ainvoke(Command(resume=user_text), config=config)

        elif state.values:
            # Previous session completed — auto-start fresh to avoid broken
            # message history (orphaned tool_call_ids → OpenAI 400 error)
            reset_counter += 1
            context.user_data["reset_counter"] = reset_counter
            config = {"configurable": {"thread_id": f"tg_{chat_id}_r{reset_counter}"}}
            logger.info("Previous session complete, rotating thread", extra={"chat_id": chat_id})
            await update.message.reply_text(
                "✨ Phiên nghiên cứu trước đã hoàn thành. Bắt đầu nghiên cứu mới..."
            )
            coro = agent.ainvoke(
                {"messages": [HumanMessage(content=user_text)]}, config=config
            )

        else:
            # Brand-new session
            coro = agent.ainvoke(
                {"messages": [HumanMessage(content=user_text)]}, config=config
            )

        result = await asyncio.wait_for(coro, timeout=REQUEST_TIMEOUT_SECONDS)

        # Check if graph paused again (more clarification needed)
        new_state = await agent.aget_state(config)
        if new_state.next:
            question = new_state.tasks[0].interrupts[0].value
            await update.message.reply_text(f"❓ {question}")
        elif "final_report" in result:
            report = result["final_report"]
            logger.info("Research complete", extra={"chat_id": chat_id, "length": len(report)})
            for i in range(0, len(report), 4096):
                await update.message.reply_text(report[i : i + 4096])
        else:
            messages = result.get("messages", [])
            if messages:
                await update.message.reply_text(messages[-1].content)

    except asyncio.TimeoutError:
        logger.error("Request timed out", extra={"chat_id": chat_id})
        await update.message.reply_text(
            f"⏱️ Quá thời gian xử lý ({REQUEST_TIMEOUT_SECONDS}s). Vui lòng thử lại với câu hỏi đơn giản hơn."
        )
    except Exception:
        logger.exception("Error handling message", extra={"chat_id": chat_id})
        await update.message.reply_text("❌ Có lỗi xảy ra. Vui lòng thử lại.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset conversation — /reset command."""
    chat_id = str(update.effective_chat.id)
    counter = context.user_data.get("reset_counter", 0) + 1
    context.user_data["reset_counter"] = counter
    logger.info("Session reset", extra={"chat_id": chat_id, "counter": counter})
    await update.message.reply_text("🔄 Đã reset. Gửi câu hỏi mới để bắt đầu.")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN not found in .env")
        return

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
