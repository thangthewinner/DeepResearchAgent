import asyncio

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown

from src.graph import build_main_agent

console = Console()


async def main():
    # Build agent
    agent = build_main_agent()

    # Simple query
    query = "What are the top 3 semiconductor companies in 2024?"

    console.print(f"\n🔍 [bold cyan]Query:[/bold cyan] {query}\n")

    # Run agent
    config = {"configurable": {"thread_id": "simple_1"}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]}, config=config
    )

    # Display result
    console.print("\n📄 [bold green]Final Report:[/bold green]\n")
    console.print(Markdown(result["final_report"]))


if __name__ == "__main__":
    asyncio.run(main())
