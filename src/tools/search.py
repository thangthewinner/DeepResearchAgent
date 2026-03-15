from typing import Annotated, List, Literal

from config.settings import MAX_CONTEXT_LENGTH, TAVILY_API_KEY
from langchain_core.tools import InjectedToolArg, tool
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tavily import TavilyClient  # type: ignore[import-untyped]

from ..logging_config import get_logger
from ..utils.summarization import summarize_webpage_content

logger = get_logger(__name__)

TAVILY_RESULTS_KEY = "results"

SearchResult = dict[str, str]
SearchResponse = dict[str, list[SearchResult]]

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def _log_retry(retry_state: RetryCallState) -> None:
    """Log a warning before a retry attempt."""
    exception = (
        retry_state.outcome.exception()
        if retry_state.outcome and retry_state.outcome.failed
        else None
    )
    logger.warning(
        "Tavily search retry",
        extra={
            "attempt": retry_state.attempt_number,
            "error": str(exception),
        },
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=_log_retry,
)
def _tavily_search_with_retry(
    query: str,
    max_results: int,
    topic: str,
    include_raw_content: bool,
) -> SearchResponse:
    """Wrapped API call to enable retries on a per-query basis."""
    raw_response = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    if not isinstance(raw_response, dict):
        return {TAVILY_RESULTS_KEY: []}
    results = raw_response.get(TAVILY_RESULTS_KEY, [])
    if not isinstance(results, list):
        return {TAVILY_RESULTS_KEY: []}
    normalized_results: list[SearchResult] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        normalized_results.append(
            {
                "url": str(item.get("url", "")),
                "title": str(item.get("title", "")),
                "content": str(item.get("content", "")),
                "raw_content": str(item.get("raw_content", "")),
            }
        )
    return {TAVILY_RESULTS_KEY: normalized_results}


def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> list[SearchResponse]:
    """Perform search using Tavily API."""
    logger.info("Executing Tavily search", extra={"queries": search_queries})
    search_docs: list[SearchResponse] = []
    for query in search_queries:
        try:
            result = _tavily_search_with_retry(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )
            search_docs.append(result)
        except Exception:
            logger.exception(
                "Failed to fetch Tavily results",
                extra={"query": query},
            )
    return search_docs


def deduplicate_search_results(
    search_results: list[SearchResponse],
) -> dict[str, SearchResult]:
    """Deduplicate search results by URL."""
    unique_results: dict[str, SearchResult] = {}
    for response in search_results:
        results = response.get(TAVILY_RESULTS_KEY, [])
        for result in results:
            url = result.get("url", "")
            if not url:
                continue
            if url not in unique_results:
                unique_results[url] = result
    return unique_results


def process_search_results(
    unique_results: dict[str, SearchResult],
) -> dict[str, dict[str, str]]:
    """Process and summarize search results."""
    summarized_results: dict[str, dict[str, str]] = {}
    for url, result in unique_results.items():
        raw_content = result.get("raw_content", "")
        if raw_content:
            content = summarize_webpage_content(raw_content[:MAX_CONTEXT_LENGTH])
        else:
            content = result.get("content", "")
        summarized_results[url] = {
            "title": result.get("title") or url,
            "content": content,
        }
    return summarized_results


def format_search_output(summarized_results: dict[str, dict[str, str]]) -> str:
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
