"""
LangGraph workflow definition for AMPilot research assistant.

This module defines the agent workflow using LangGraph, including:
- Tool definitions for AMP information retrieval
- LLM configuration with OpenRouter
- Graph nodes for agent reasoning and tool execution
- Conditional routing logic for tool usage
"""

from configuration import CHAT_MODEL, OPENROUTER_API_KEY
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from retrieval import retrieve_amp_info
from retrieval_graph.state import AgentState

# 1. Define Tools and ToolNode
# Available tools for the agent to use during conversations
tools = [retrieve_amp_info]
tool_node = ToolNode(tools)

# 2. Define the Model
# Configure LLM with OpenRouter API for accessing various models
llm = ChatOpenAI(
    model=CHAT_MODEL,  # Model to use (default: gpt-4o-mini)
    temperature=0,  # Deterministic responses for research accuracy
    streaming=True,  # Enable real-time token streaming
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API endpoint
    api_key=OPENROUTER_API_KEY,  # API authentication
)
# Bind tools to the LLM so it can decide when to use them
llm_with_tools = llm.bind_tools(tools)


# 3. Define Graph Nodes
def agent_node(state: AgentState):
    """
    Agent reasoning node that invokes the LLM to decide the next action.

    Args:
        state: Current conversation state with message history

    Returns:
        Updated state with LLM response
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Conditional routing function to determine workflow direction.

    Args:
        state: Current conversation state

    Returns:
        "continue" if tools should be called, "end" if conversation is complete
    """
    last_message = state["messages"][-1]
    # If the LLM wants to call tools, continue to tool execution
    if not last_message.tool_calls:
        return "end"
    return "continue"


# 4. Build the Graph
# Create the main workflow graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("agent", agent_node)  # LLM reasoning node
workflow.add_node("tools", tool_node)  # Tool execution node

# Set the entry point of the workflow
workflow.set_entry_point("agent")

# Add conditional edges for dynamic routing
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},  # Route based on tool usage decision
)

# Add edge from tools back to agent for continued conversation
workflow.add_edge("tools", "agent")

# Compile the graph into a runnable application
app = workflow.compile()
