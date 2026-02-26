from typing import Annotated, List, Literal

from config.settings import MAX_CONTEXT_LENGTH, TAVILY_API_KEY
from langchain_core.tools import InjectedToolArg, tool
from tavily import TavilyClient

from ..utils.summarization import summarize_webpage_content

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:
    """Perform search using Tavily API."""
    print(f"🔍 Executing Tavily search for: {search_queries}")
    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        search_docs.append(result)
    return search_docs


def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL."""
    unique_results = {}
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result
    return unique_results


def process_search_results(unique_results: dict) -> dict:
    """Process and summarize search results."""
    summarized_results = {}
    for url, result in unique_results.items():
        if result.get("raw_content"):
            content = summarize_webpage_content(
                result["raw_content"][:MAX_CONTEXT_LENGTH]
            )
        else:
            content = result["content"]
        summarized_results[url] = {"title": result["title"], "content": content}
    return summarized_results


def format_search_output(summarized_results: dict) -> str:
    """Format final search results."""
    if not summarized_results:
        return "No valid search results found."

    formatted_output = "Search results:\n\n"
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"
    return formatted_output


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """
    Fetch results from Tavily search API with content summarization.

    Args:
        query: A single, specific search query to execute.
        max_results: Maximum number of results to return.
        topic: Topic filter ('general', 'news', 'finance').

    Returns:
        Formatted string of deduplicated and summarized results.
    """
    search_results = tavily_search_multiple(
        [query], max_results=max_results, topic=topic, include_raw_content=True
    )
    unique_results = deduplicate_search_results(search_results)
    summarized_results = process_search_results(unique_results)
    return format_search_output(summarized_results)
