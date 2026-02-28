import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

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

from src.graph import build_main_agent

# Build agent once at startup
agent = build_main_agent()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Xin chào! Gửi câu hỏi nghiên cứu cho tôi, tôi sẽ phân tích và trả lời chi tiết."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = str(update.effective_chat.id)
    config = {"configurable": {"thread_id": f"tg_{chat_id}"}}

    await update.message.reply_text("🔍 Đang xử lý... Vui lòng chờ.")

    try:
        # Check if graph is paused (waiting for clarification answer)
        state = agent.get_state(config)

        if state.next:
            # Graph is paused at interrupt() — resume with user's answer
            result = await agent.ainvoke(
                Command(resume=user_text), config=config
            )
        else:
            # No paused graph — start a new research session
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_text)]}, config=config
            )

        # Check if graph paused again (needs more clarification)
        new_state = agent.get_state(config)

        if new_state.next:
            # Graph is paused — the interrupt value is the clarification question
            # It's stored in state.tasks tuple
            question = new_state.tasks[0].interrupts[0].value
            await update.message.reply_text(f"❓ {question}")
        elif "final_report" in result:
            report = result["final_report"]
            # Telegram has 4096 char limit per message
            for i in range(0, len(report), 4096):
                await update.message.reply_text(report[i : i + 4096])
        else:
            messages = result.get("messages", [])
            if messages:
                await update.message.reply_text(messages[-1].content)

    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset conversation for this chat — /reset command."""
    await update.message.reply_text("🔄 Đã reset. Gửi câu hỏi mới để bắt đầu.")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env")
        return

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Telegram bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
