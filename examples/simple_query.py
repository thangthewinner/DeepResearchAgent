import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from config.settings import REQUEST_TIMEOUT_SECONDS
from src.logging_config import get_logger
from src.runtime import bootstrap_agent

agent = bootstrap_agent()
logger = get_logger(__name__)
console = Console()


async def main():
    query = "Most notable AI projects in 2026 so far"
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")
    logger.info("Starting research", extra={"query": query})

    config = {"configurable": {"thread_id": "simple_1"}}

    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=config),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        console.print(
            f"\n[bold red]Timed out after {REQUEST_TIMEOUT_SECONDS}s[/bold red]\n"
        )
        return

    # Handle clarification loop
    state = await agent.aget_state(config)
    while state.next:
        question = state.tasks[0].interrupts[0].value
        console.print(
            f"\n[bold yellow]Clarification needed:[/bold yellow] {question}\n"
        )
        user_input = input("Your answer: ")

        try:
            result = await asyncio.wait_for(
                agent.ainvoke(Command(resume=user_input), config=config),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            console.print(
                f"\n[bold red]Timed out after {REQUEST_TIMEOUT_SECONDS}s[/bold red]\n"
            )
            return

        state = await agent.aget_state(config)

    # Display result
    if "final_report" in result:
        console.print("\n[bold green]Final Report:[/bold green]\n")
        console.print(Markdown(result["final_report"]))
        logger.info("Research complete", extra={"length": len(result["final_report"])})
    else:
        console.print("\n[bold yellow]No final report generated.[/bold yellow]\n")
        messages = result.get("messages", [])
        if messages:
            console.print(messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
