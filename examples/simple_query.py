import asyncio
import os
import sys
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from config.settings import REQUEST_TIMEOUT_SECONDS
from src.graph import build_main_agent
from src.logging_config import get_logger, setup_logging

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

setup_logging()
logger = get_logger(__name__)
console = Console()

# MemorySaver by default — no persistence needed for CLI usage
agent = build_main_agent()


async def main():
    query = "Dự án nổi bật về AI trong năm 2026 tính đến hiện nay"
    console.print(f"\n🔍 [bold cyan]Query:[/bold cyan] {query}\n")
    logger.info("Starting research", extra={"query": query})

    config = {"configurable": {"thread_id": "simple_1"}}

    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=config),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        console.print(f"\n⏱️ [bold red]Timed out after {REQUEST_TIMEOUT_SECONDS}s[/bold red]\n")
        return

    # Handle clarification loop
    state = await agent.aget_state(config)
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

        state = await agent.aget_state(config)

    # Display result
    if "final_report" in result:
        console.print("\n📄 [bold green]Final Report:[/bold green]\n")
        console.print(Markdown(result["final_report"]))
        logger.info("Research complete", extra={"length": len(result["final_report"])})
    else:
        console.print("\n⚠️ [bold yellow]No final report generated.[/bold yellow]\n")
        messages = result.get("messages", [])
        if messages:
            console.print(messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
