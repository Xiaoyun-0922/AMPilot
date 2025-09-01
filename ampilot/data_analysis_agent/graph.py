from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode # 导入 ToolNode

class AgentState(TypedDict):
    """
    Represents the state of our agent. It's a list of messages that is appended to over time.
    """
    messages: Annotated[list, lambda x, y: x + y]

# 注意函数签名，它现在接收一个 'tools' 列表
def create_workflow(llm_with_tools, tools):
    """
    Builds and compiles the LangGraph agent workflow using the builder pattern and ToolNode.
    """
    def call_model(state):
        """Node that invokes the LLM."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 我们不再需要手写的 call_tool 函数了！

    def should_continue(state):
        """
        Conditional edge: decides whether to continue calling tools or end the process.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    # --- Graph Construction using the Builder Pattern ---
    
    builder = StateGraph(AgentState)

    builder.add_node("agent", call_model)
    
    # 直接使用 ToolNode 创建 action 节点
    tool_node = ToolNode(tools)
    builder.add_node("action", tool_node)

    builder.add_edge("__start__", "agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    builder.add_edge("action", "agent")

    app = builder.compile()
    
    return app