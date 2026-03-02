import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from config.settings import REQUEST_TIMEOUT_SECONDS, SQLITE_DB_PATH
from src.graph import build_main_agent
from src.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
console = Console()


async def main():
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as checkpointer:
        agent = build_main_agent(checkpointer)

        query = "Dự án nổi bật về AI trong năm 2026 tính đến hiện nay"
        console.print(f"\n🔍 [bold cyan]Query:[/bold cyan] {query}\n")
        logger.info("Starting research", extra={"query": query})

        config = {"configurable": {"thread_id": "simple_1"}}

        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=query)]}, config=config
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            console.print(f"\n⏱️ [bold red]Timed out after {REQUEST_TIMEOUT_SECONDS}s[/bold red]\n")
            return

        # Handle clarification loop
        state = agent.get_state(config)
        while state.next:
            question = state.tasks[0].interrupts[0].value
            console.print(f"\n❓ [bold yellow]Clarification needed:[/bold yellow] {question}\n")
            user_input = input("Your answer: ")

            try:
                result = await asyncio.wait_for(
                    agent.ainvoke(Command(resume=user_input), config=config),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                console.print(f"\n⏱️ [bold red]Timed out after {REQUEST_TIMEOUT_SECONDS}s[/bold red]\n")
                return

            state = agent.get_state(config)

        # Display result
        if "final_report" in result:
            console.print("\n📄 [bold green]Final Report:[/bold green]\n")
            console.print(Markdown(result["final_report"]))
            logger.info("Research complete", extra={"report_length": len(result["final_report"])})
        else:
            console.print("\n⚠️ [bold yellow]No final report generated.[/bold yellow]\n")
            messages = result.get("messages", [])
            if messages:
                console.print(messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
