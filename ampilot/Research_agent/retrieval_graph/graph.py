from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

# Import configuration and tools
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration import OPENAI_API_KEY
from retrieval import find_peptide_properties, find_peptides_by_properties
from router import route_query
from retrieval_graph.prompts import SYSTEM_PROMPT, SEQUENCE_TO_PROPERTIES_PROMPT, PROPERTIES_TO_SEQUENCE_PROMPT, FINALIZE_PROMPT

class AgentState(TypedDict):
    """
    Enhanced state for the AMPilot agent with routing capabilities.
    """
    messages: Annotated[list, lambda x, y: x + y]
    query_intent: str  # "sequence_to_properties", "properties_to_sequence", or "general_conversation"
    extracted_info: dict  # Information extracted by the router
    current_step: str  # Current processing step

def create_workflow():
    """
    Creates the enhanced AMPilot workflow with routing capabilities.
    """

    # Initialize LLM
    llm = initialize_llm()

    # Define all available tools
    all_tools = [route_query, find_peptide_properties, find_peptides_by_properties]

    # Create tool nodes
    router_tool = ToolNode([route_query])
    sequence_tool = ToolNode([find_peptide_properties])
    properties_tool = ToolNode([find_peptides_by_properties])

    def router_node(state):
        """Route the user query to determine intent."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, 'content'):
            user_query = last_message.content
        else:
            user_query = str(last_message)

        # Use router to classify the query via LLM tool-calling
        llm_with_router = llm.bind_tools([route_query])

        # Add system message for routing
        routing_messages = [
            SystemMessage(content="You are a query router. Use the route_query tool to classify the user's intent."),
            last_message
        ]

        response = llm_with_router.invoke(routing_messages)

        return {
            "messages": [response],
            "current_step": "routing"
        }

    def sequence_to_properties_node(state):
        """Handle sequence-to-properties queries."""
        messages = state["messages"]

        # Create specialized LLM for sequence analysis
        llm_with_seq_tools = llm.bind_tools([find_peptide_properties])

        # Add specialized system prompt
        specialized_messages = [
            SystemMessage(content=SEQUENCE_TO_PROPERTIES_PROMPT)
        ] + messages

        response = llm_with_seq_tools.invoke(specialized_messages)

        return {
            "messages": [response],
            "current_step": "sequence_analysis"
        }

    def properties_to_sequence_node(state):
        """Handle properties-to-sequence queries."""
        messages = state["messages"]

        # Create specialized LLM for property-based search
        llm_with_prop_tools = llm.bind_tools([find_peptides_by_properties])

        # Add specialized system prompt
        specialized_messages = [
            SystemMessage(content=PROPERTIES_TO_SEQUENCE_PROMPT)
        ] + messages

        response = llm_with_prop_tools.invoke(specialized_messages)

        return {
            "messages": [response],
            "current_step": "property_search"
        }

    def general_conversation_node(state):
        """Handle general conversation."""
        messages = state["messages"]

        # Extract last user query
        user_query = ""
        for m in reversed(messages):
            # Tuple format ("human", text)
            if isinstance(m, tuple) and len(m) == 2 and m[0] == "human":
                user_query = str(m[1])
                break
            # Message object with content
            if hasattr(m, 'type') and getattr(m, 'type', None) == 'human' and hasattr(m, 'content'):
                user_query = str(m.content)
                break
            if hasattr(m, 'content') and not hasattr(m, 'tool_calls'):
                # Best-effort fallback
                user_query = str(m.content)
                break

        # Heuristic: detect if query is AMP-related
        def is_amp_related(text: str) -> bool:
            if not text:
                return False
            t = text.lower()
            keywords = [
                'antimicrobial', 'peptide', 'amp', 'bacteria', 'bacterium', 'mic', 'minimum inhibitory',
                'sequence', 'amino acid', 'gram-positive', 'gram negative', 'weaviate', 'dbaasp', 'dramp', 'grampa'
            ]
            if any(k in t for k in keywords):
                return True
            import re
            # AA sequence pattern
            if re.search(r'[ACDEFGHIKLMNPQRSTVWY]{10,}', text):
                return True
            return False

        if not is_amp_related(user_query):
            guidance_html = (
                "<div class=\"amp-analysis\">"
                "<div class=\"peptide-header\"><h3>🧬 AMPilot</h3>"
                "<p class=\"summary\">I focus on antimicrobial peptides (AMPs). Please ask AMP-related questions.</p>"
                "</div>"
                "<div class=\"antimicrobial-profile\">"
                "<h4>💡 How I can help</h4>"
                "<ul>"
                "<li>Give a peptide sequence to analyze its activity (bacteria, MIC, modifications)</li>"
                "<li>Find peptides active against a bacterium (e.g., S. aureus) with MIC range</li>"
                "<li>Search peptides with specific modifications (e.g., disulfide bonds, amidation)</li>"
                "</ul>"
                "</div>"
                "<div><strong>Examples:</strong> What is the principle of GFGCPGDAYQCSEHCRALGGGRTGGYCAGPWYLGHPTCTCS?</div>"
                "</div>"
            )
            return {"messages": [AIMessage(content=guidance_html)], "current_step": "conversation"}

        # Otherwise, continue with normal LLM response
        general_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm.invoke(general_messages)
        return {"messages": [response], "current_step": "conversation"}


    def finalize_node(state):
        """Finalize: collapse prior content into ONE polished user-facing message."""
        messages = state["messages"]
        final_messages = [SystemMessage(content=FINALIZE_PROMPT)] + messages
        response = llm.invoke(final_messages)
        return {"messages": [response], "current_step": "finalize"}

    def tool_execution_node(state):
        """Execute the appropriate tools based on the current step."""
        messages = state["messages"]
        last_message = messages[-1]
        current_step = state.get("current_step", "")

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Determine which tool node to use based on the tool being called
            tool_name = last_message.tool_calls[0].get('name', '')

            if tool_name == 'route_query':
                return router_tool.invoke({"messages": messages})
            elif tool_name == 'find_peptide_properties':
                return sequence_tool.invoke({"messages": messages})
            elif tool_name == 'find_peptides_by_properties':
                return properties_tool.invoke({"messages": messages})

        return {"messages": messages}

    def route_after_classification(state):
        """Route to appropriate specialized node after classification."""
        messages = state["messages"]

        # Look for routing result in the last tool message
        intent = "general_conversation"
        routing_message_index = None

        for i, message in enumerate(reversed(messages)):
            if hasattr(message, 'content') and 'intent' in str(message.content):
                try:
                    import json
                    if isinstance(message.content, str):
                        content = json.loads(message.content)
                    else:
                        content = message.content

                    intent = content.get('intent', 'general_conversation')
                    routing_message_index = len(messages) - 1 - i  # Convert to forward index
                    break
                except:
                    pass

        # Remove the routing message from the state to prevent it from appearing in final output
        if routing_message_index is not None:
            # Keep only the human message and remove the routing tool message
            filtered_messages = []
            for i, msg in enumerate(messages):
                if i != routing_message_index:  # Skip the routing message
                    filtered_messages.append(msg)
            state["messages"] = filtered_messages

        if intent == 'sequence_to_properties':
            return "sequence_analysis"
        elif intent == 'properties_to_sequence':
            return "property_search"
        else:
            return "general_conversation"

    def should_continue_with_tools(state):
        """Decide whether to continue with tool execution."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "execute_tools"
        else:
            return "end"

    # Build the graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("sequence_analysis", sequence_to_properties_node)
    builder.add_node("property_search", properties_to_sequence_node)
    builder.add_node("general_conversation", general_conversation_node)
    builder.add_node("execute_tools", tool_execution_node)
    builder.add_node("finalize", finalize_node)

    # Add edges
    builder.add_edge("__start__", "router")

    # Conditional routing after classification
    builder.add_conditional_edges(
        "router",
        should_continue_with_tools,
        {
            "execute_tools": "execute_tools",
            "end": "finalize"
        }
    )

    # After tool execution, route to appropriate handler
    builder.add_conditional_edges(
        "execute_tools",
        route_after_classification,
        {
            "sequence_analysis": "sequence_analysis",
            "property_search": "property_search",
            "general_conversation": "general_conversation"
        }
    )

    # Handle tool calls in specialized nodes
    builder.add_conditional_edges(
        "sequence_analysis",
        should_continue_with_tools,
        {
            "execute_tools": "execute_tools",
            "end": "finalize"
        }
    )

    builder.add_conditional_edges(
        "property_search",
        should_continue_with_tools,
        {
            "execute_tools": "execute_tools",
            "end": "finalize"
        }
    )

    builder.add_edge("general_conversation", "finalize")
    builder.add_edge("finalize", END)

    # Add memory
    memory = MemorySaver()
    app = builder.compile(checkpointer=memory)

    return app

# Initialize the LLM with OpenAI API
def initialize_llm():
    """Initialize ChatOpenAI with OpenAI configuration."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

    # 使用OpenAI API
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 使用OpenAI的模型
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
        # 移除OpenRouter相关配置，使用默认的OpenAI API
    )
    return llm

# Create the main application
def create_app():
    """Create and return the compiled LangGraph application."""
    return create_workflow()

# Create the global app instance
app = create_app()