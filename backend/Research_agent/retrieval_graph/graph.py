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

from configuration import OPENAI_API_KEY, CHAT_MODEL
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
        # Force the router to actually call the routing tool
        llm_with_router = llm.bind_tools([route_query], tool_choice="required")

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
        # Force tool usage to avoid generic LLM answers here
        llm_with_seq_tools = llm.bind_tools([find_peptide_properties], tool_choice="required")

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
        # Force tool usage to avoid generic LLM answers here
        llm_with_prop_tools = llm.bind_tools([find_peptides_by_properties], tool_choice="required")

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
        """Finalize with LLM-enhanced formatting for better presentation."""
        import json as _json
        messages = state["messages"]
        intent = state.get("query_intent", "")

        # Find the last tool result message
        tool_payload = None
        tool_name = None
        for m in reversed(messages):
            if getattr(m, "type", None) == "tool" and hasattr(m, "content"):
                try:
                    tool_payload = _json.loads(m.content)
                    tool_name = getattr(m, "name", None)
                    break
                except Exception:
                    continue

        if not isinstance(tool_payload, list):
            # Fallback minimal message if no tool payload found
            text = "No structured results were returned by the retrieval tools."
            return {"messages": [AIMessage(content=text)], "current_step": "finalize"}

        # Check if we have any actual results (not just error messages)
        valid_results = [r for r in tool_payload if not r.get("message") and not r.get("error")]

        if not valid_results:
            # Handle no results case
            error_msg = tool_payload[0].get("message", "No matching peptides found.")
            return {"messages": [AIMessage(content=error_msg)], "current_step": "finalize"}

        # Get the original user query for context
        user_query = ""
        for m in messages:
            if hasattr(m, 'content') and not hasattr(m, 'tool_calls') and getattr(m, 'type', None) != 'tool':
                user_query = m.content
                break

        # Create structured data summary for LLM
        data_summary = _format_results_for_llm(valid_results, intent, tool_name)

        # Use LLM to create a well-formatted response
        formatting_prompt = f"""Based on the user query: "{user_query}"

Here are the search results from the GRAMPA antimicrobial peptide database:

{data_summary}

CRITICAL INSTRUCTIONS:
- You MUST include ALL {len(valid_results)} peptides in your response
- Do NOT skip, truncate, or summarize any peptides
- Present each peptide with complete information
- Number them sequentially from 1 to {len(valid_results)}
- Even if some sequences appear similar or identical, include ALL entries as separate peptides
- Each peptide entry in the data represents a unique database record and must be shown

Please provide a well-formatted, professional response that:
1. Directly answers the user's question
2. Presents ALL {len(valid_results)} peptides in a clear, organized manner
3. Uses proper scientific formatting with numbered sections for each peptide
4. Highlights key findings and patterns at the end
5. Only mentions information that is actually present in the data
6. Uses markdown formatting for better readability

Format each peptide EXACTLY as:
**Peptide X**
- **Sequence:** [sequence]
- **Length:** [length] amino acids
- **Target Bacterium:** [bacterium]
- **MIC Value:** [mic] µM
- **Modifications:** [modifications]
- **Source:** [source link]

IMPORTANT: The data contains {len(valid_results)} separate entries. You must create exactly {len(valid_results)} peptide sections, numbered 1 through {len(valid_results)}. Do not combine or omit any entries."""

        # Create a completely deterministic formatted response
        formatted_response = f"""Based on the user query: "{user_query}"

Here are the search results from the GRAMPA antimicrobial peptide database:

"""

        # Debug: Print what we're formatting
        print(f"DEBUG: Formatting {len(valid_results)} results for LLM")
        gigk_found_in_formatting = False
        for i, result in enumerate(valid_results, 1):
            seq = result.get("sequence", "N/A")
            print(f"  {i}. {seq} (len {len(seq) if seq != 'N/A' else 'N/A'})")
            if seq == "GIGKFLKKAKKFGKAFVKILKK":
                gigk_found_in_formatting = True
                print(f"      *** GIGKFLKKAKKFGKAFVKILKK FOUND IN FORMATTING! ***")

        if not gigk_found_in_formatting:
            print("  ❌ GIGKFLKKAKKFGKAFVKILKK NOT FOUND IN FORMATTING!")
        else:
            print("  ✅ GIGKFLKKAKKFGKAFVKILKK FOUND IN FORMATTING!")

        # Add each peptide deterministically
        for i, result in enumerate(valid_results, 1):
            seq = result.get("sequence", "N/A")
            length = len(seq) if isinstance(seq, str) and seq != "N/A" else "Unknown"
            mic = result.get("mic_value_um")
            mic_str = f"{mic} µM" if mic is not None else "Not available"
            bact = result.get("bacterium", "Unknown")
            mods = result.get("modifications", [])
            mods_str = str(mods) if mods else "[]"

            # Get source URL
            source_url = result.get("url") or result.get("url_source") or result.get("database") or "Unknown"
            if source_url.startswith("http"):
                source_link = f"[Link]({source_url})"
            else:
                source_link = f"[Link](#{source_url})"

            formatted_response += f"""**Peptide {i}**
- **Sequence:** {seq}
- **Length:** {length} amino acids
- **Target Bacterium:** {bact}
- **MIC Value:** {mic_str}
- **Modifications:** {mods_str}
- **Source:** {source_link}

"""

        # Add summary
        formatted_response += f"""### Key Findings and Patterns
- All peptides target the specified bacterium with varying MIC values.
- The peptides show different levels of antimicrobial effectiveness.
- Some peptides have modifications that may influence their activity.
- MIC values indicate the minimum concentration needed to inhibit bacterial growth."""

        response = AIMessage(content=formatted_response)
        return {"messages": [response], "current_step": "finalize"}

    def _format_results_for_llm(results, intent, tool_name):
        """Format search results into a structured summary for LLM processing."""
        def source_of(d):
            return d.get("url") or d.get("url_source") or d.get("database") or d.get("source") or "Unknown"

        formatted_lines = []

        if intent == "sequence_to_properties" or tool_name == "find_peptide_properties":
            # Format for sequence analysis
            seq = results[0].get("sequence", "") if results else ""
            formatted_lines.append(f"SEQUENCE ANALYSIS FOR: {seq}")
            formatted_lines.append("=" * 50)

            for i, r in enumerate(results, 1):
                mic = r.get("mic_value_um")
                mic_str = f"{mic} µM" if mic is not None else "Not available"
                bact = r.get("bacterium", "Unknown")
                strain = r.get("strain", "") or ""
                mods = r.get("modifications", "") or "None"
                src = source_of(r)

                formatted_lines.append(f"\nResult {i}:")
                formatted_lines.append(f"  Target Bacterium: {bact}")
                if strain:
                    formatted_lines.append(f"  Bacterial Strain: {strain}")
                formatted_lines.append(f"  MIC Value: {mic_str}")
                formatted_lines.append(f"  Modifications: {mods}")
                formatted_lines.append(f"  Source: {src}")
        else:
            # Format for peptide search results
            formatted_lines.append("PEPTIDE SEARCH RESULTS")
            formatted_lines.append("=" * 50)

            for i, r in enumerate(results, 1):
                seq = r.get("sequence", "N/A")
                length = len(seq) if isinstance(seq, str) and seq != "N/A" else "Unknown"
                mic = r.get("mic_value_um")
                mic_str = f"{mic} µM" if mic is not None else "Not available"
                bact = r.get("bacterium", "Unknown")
                mods = r.get("modifications", "") or "None"
                src = source_of(r)

                # Special emphasis for important sequences
                emphasis = ""
                if seq == "GIGKFLKKAKKFGKAFVKILKK":
                    emphasis = " *** IMPORTANT SEQUENCE - MUST INCLUDE ***"

                formatted_lines.append(f"\nPeptide {i}:{emphasis}")
                formatted_lines.append(f"  Sequence: {seq}")
                formatted_lines.append(f"  Length: {length} amino acids")
                formatted_lines.append(f"  Target Bacterium: {bact}")
                formatted_lines.append(f"  MIC Value: {mic_str}")
                formatted_lines.append(f"  Modifications: {mods}")
                formatted_lines.append(f"  Source: {src}")

        return "\n".join(formatted_lines)

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
        """Decide next node after executing a tool."""
        messages = state["messages"]

        # Identify the last executed tool, if any
        last_tool_name = None
        for i, message in enumerate(reversed(messages)):
            if getattr(message, "type", None) == "tool":
                last_tool_name = getattr(message, "name", "") or ""
                break

        # If we just executed a retrieval tool, go straight to finalize
        if last_tool_name in ("find_peptide_properties", "find_peptides_by_properties"):
            return "finalize"

        # Otherwise, we expect we just executed the router tool; extract intent
        intent = "general_conversation"
        routing_message_index = None

        for i, message in enumerate(reversed(messages)):
            # Check for tool message with route_query result
            if hasattr(message, 'content') and isinstance(message.content, str):
                try:
                    import json
                    content = json.loads(message.content)
                    if isinstance(content, dict) and 'intent' in content:
                        intent = content.get('intent', 'general_conversation')
                        routing_message_index = len(messages) - 1 - i
                        break
                except:
                    pass

            # Check for ToolMessage type
            if getattr(message, 'type', None) == 'tool':
                try:
                    if hasattr(message, 'content'):
                        import json
                        content = json.loads(message.content)
                        if isinstance(content, dict) and 'intent' in content:
                            intent = content.get('intent', 'general_conversation')
                            routing_message_index = len(messages) - 1 - i
                            break
                except:
                    pass

        # Store intent in state for debugging
        state["query_intent"] = intent

        # Clean up routing messages to prevent them from appearing in final output
        if routing_message_index is not None:
            filtered_messages = []
            for i, msg in enumerate(messages):
                # Keep human messages and remove routing tool messages
                if i != routing_message_index:
                    filtered_messages.append(msg)
            state["messages"] = filtered_messages

        print(f"DEBUG: Routing intent determined as: {intent}")

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
            "general_conversation": "general_conversation",
            "finalize": "finalize"
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

    # Using openai API
    llm = ChatOpenAI(
    # You can also choose another model, there is no high requirement for LLM's ability.
        model=CHAT_MODEL,
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
    )
    return llm

# Create the main application
def create_app():
    """Create and return the compiled LangGraph application."""
    return create_workflow()

# Create the global app instance
app = create_app()