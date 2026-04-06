from src.tools.registry import get_researcher_tools


def test_registry_returns_native_tools() -> None:
    """Test that registry returns the native researcher tools."""
    tool_names = [tool.name for tool in get_researcher_tools()]

    assert tool_names == ["tavily_search", "think_tool"]
