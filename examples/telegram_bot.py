import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config.settings import REQUEST_TIMEOUT_SECONDS, SQLITE_DB_PATH
from src.graph import build_main_agent
from src.logging_config import get_logger, setup_logging
from src.server import run_server

setup_logging()
logger = get_logger(__name__)

# Keep a strong reference to the background task to prevent garbage collection
background_tasks = set()


async def post_init(application: Application) -> None:
    """Lifecycle hook: runs after bot starts. Sets up SQLite checkpointer and agent."""
    conn = await asyncio.get_event_loop().run_in_executor(None, lambda: None)
    checkpointer = await AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH).__aenter__()
    application.bot_data["checkpointer"] = checkpointer
    application.bot_data["agent"] = build_main_agent(checkpointer)
    logger.info("Agent initialized with SQLite checkpointer", extra={"db": SQLITE_DB_PATH})

    # Start the FastAPI health check server in the background
    task = asyncio.create_task(run_server())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


async def post_shutdown(application: Application) -> None:
    """Lifecycle hook: runs before bot exits. Closes SQLite connection."""
    checkpointer = application.bot_data.get("checkpointer")
    if checkpointer:
        await checkpointer.__aexit__(None, None, None)
        logger.info("SQLite checkpointer closed")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Xin chào! Gửi câu hỏi nghiên cứu cho tôi, tôi sẽ phân tích và trả lời chi tiết."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = str(update.effective_chat.id)
    reset_counter = context.user_data.get("reset_counter", 0)
    config = {"configurable": {"thread_id": f"tg_{chat_id}_r{reset_counter}"}}
    agent = context.application.bot_data["agent"]

    logger.info("Received message", extra={"chat_id": chat_id, "text_length": len(user_text)})
    await update.message.reply_text("🔍 Đang xử lý... Vui lòng chờ.")

    try:
        state = agent.get_state(config)

        if state.next:
            # Graph is paused at interrupt() — resume with user's answer
            coro = agent.ainvoke(Command(resume=user_text), config=config)
        else:
            # New research session
            coro = agent.ainvoke(
                {"messages": [HumanMessage(content=user_text)]}, config=config
            )

        result = await asyncio.wait_for(coro, timeout=REQUEST_TIMEOUT_SECONDS)

        # Check if graph paused again (needs more clarification)
        new_state = agent.get_state(config)
        if new_state.next:
            question = new_state.tasks[0].interrupts[0].value
            await update.message.reply_text(f"❓ {question}")
        elif "final_report" in result:
            report = result["final_report"]
            logger.info("Research complete", extra={"chat_id": chat_id, "report_length": len(report)})
            # Telegram has 4096 char limit per message
            for i in range(0, len(report), 4096):
                await update.message.reply_text(report[i : i + 4096])
        else:
            messages = result.get("messages", [])
            if messages:
                await update.message.reply_text(messages[-1].content)

    except asyncio.TimeoutError:
        logger.error("Request timed out", extra={"chat_id": chat_id, "timeout": REQUEST_TIMEOUT_SECONDS})
        await update.message.reply_text(
            f"⏱️ Quá thời gian xử lý ({REQUEST_TIMEOUT_SECONDS}s). Vui lòng thử lại với câu hỏi đơn giản hơn."
        )
    except Exception as e:
        logger.exception("Error handling message", extra={"chat_id": chat_id})
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset conversation for this chat — /reset command."""
    chat_id = str(update.effective_chat.id)
    counter = context.user_data.get("reset_counter", 0) + 1
    context.user_data["reset_counter"] = counter
    logger.info("Session reset", extra={"chat_id": chat_id, "new_counter": counter})
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
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
