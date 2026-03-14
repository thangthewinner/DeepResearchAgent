from config.settings import BASE_MODEL, OPENAI_API_KEY
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

from .agents import researcher, supervisor, writer
from .models.state import (
    AgentInputState,
    AgentState,
    ResearcherOutputState,
    ResearcherState,
    SupervisorState,
)
from .nodes import clarification, context, evaluation, research
from .tools.common import ConductResearch, ResearchComplete, think_tool
from .tools.search import tavily_search

# Initialize models
base_model = init_chat_model(model=BASE_MODEL, api_key=OPENAI_API_KEY)

# Initialize supervisor tools
supervisor_tools = [
    ConductResearch,
    ResearchComplete,
    think_tool,
    writer.refine_draft_report,
]
supervisor_model_with_tools = base_model.bind_tools(supervisor_tools)

# Initialize researcher tools
researcher_tools = [tavily_search, think_tool]
model_with_tools = base_model.bind_tools(researcher_tools)


def build_researcher_agent():
    """Build researcher sub-graph."""
    agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

    agent_builder.add_node("llm_call", researcher.llm_call)
    agent_builder.add_node("tool_node", researcher.tool_node)
    agent_builder.add_node("compress_research", researcher.compress_research)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        researcher.should_continue,
        {"tool_node": "tool_node", "compress_research": "compress_research"},
    )
    agent_builder.add_edge("tool_node", "llm_call")
    agent_builder.add_edge("compress_research", END)

    return agent_builder.compile()


def build_supervisor_agent(researcher_agent):
    """Build supervisor sub-graph."""
    supervisor_builder = StateGraph(SupervisorState)

    supervisor_builder.add_node("supervisor", supervisor.supervisor_node)
    supervisor_builder.add_node(
        "supervisor_tools", supervisor.make_supervisor_tools_node(researcher_agent)
    )
    supervisor_builder.add_node("red_team", evaluation.red_team_node)
    supervisor_builder.add_node("context_pruner", context.context_pruning_node)

    supervisor_builder.add_edge(START, "supervisor")
    supervisor_builder.add_edge("supervisor", "supervisor_tools")
    supervisor_builder.add_edge("red_team", "supervisor")
    supervisor_builder.add_edge("context_pruner", "supervisor")

    return supervisor_builder.compile()



def build_main_agent():
    """Build main agent graph."""
    researcher_agent = build_researcher_agent()
    supervisor_agent = build_supervisor_agent(researcher_agent)

    builder = StateGraph(AgentState, input_schema=AgentInputState)

    builder.add_node("clarify_with_user", clarification.clarify_with_user)
    builder.add_node("write_research_brief", clarification.write_research_brief)
    builder.add_node("write_draft_report", research.write_draft_report)
    builder.add_node("supervisor_subgraph", supervisor_agent)
    builder.add_node("final_report_generation", writer.final_report_generation)

    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", "write_draft_report")
    builder.add_edge("write_draft_report", "supervisor_subgraph")
    builder.add_edge("supervisor_subgraph", "final_report_generation")
    builder.add_edge("final_report_generation", END)

    return builder.compile()
