from mcp.server.fastmcp import FastMCP
from src.tools.search import tavily_search

mcp = FastMCP("deepresearch-mcp")


@mcp.tool
def web_search(query: str, max_results: int = 3, topic: str = "general") -> str:
    """Search the web and return summarized results"""
    return tavily_search.invoke(
        {"query": query, "max_results": max_results, "topic": topic}
    )


@mcp.tool
def fetch_url(url: str) -> str:
    """Fetch a URL by searching directly for that URL as query"""
    return tavily_search.invoke({"query": url, "max_results": 1, "topic": "general"})


@mcp.resource("research://citation-policy")
def citation_policy() -> str:
    return (
        "Always cite claims with source URL and avoid unsupported factual statements."
    )


@mcp.prompt
def refine_query_prompt(topic: str) -> str:
    return (
        "Rewrite the topic into 3 specific, non-overlapping search queries. "
        f"Topic: {topic}"
    )


if __name__ == "__main__":
    mcp.run()
