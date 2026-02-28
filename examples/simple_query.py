import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from src.graph import build_main_agent

console = Console()


async def main():
    # Build agent
    agent = build_main_agent()

    # Simple query
    query = "Dự án nổi bật về AI trong năm 2026 tính đến hiện nay"

    console.print(f"\n🔍 [bold cyan]Query:[/bold cyan] {query}\n")

    # Run agent
    config = {"configurable": {"thread_id": "simple_1"}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]}, config=config
    )

    # Check if agent needs clarification (graph paused at interrupt)
    state = agent.get_state(config)
    while state.next:
        question = state.tasks[0].interrupts[0].value
        console.print(f"\n❓ [bold yellow]Clarification needed:[/bold yellow] {question}\n")
        user_input = input("Your answer: ")
        result = await agent.ainvoke(
            Command(resume=user_input), config=config
        )
        state = agent.get_state(config)

    # Display result
    if "final_report" in result:
        console.print("\n📄 [bold green]Final Report:[/bold green]\n")
        console.print(Markdown(result["final_report"]))
    else:
        console.print(
            "\n⚠️ [bold yellow]No final report generated.[/bold yellow]\n"
        )
        messages = result.get("messages", [])
        if messages:
            console.print(messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
