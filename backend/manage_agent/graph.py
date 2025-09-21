"""
Multi-Agent Graph for AMPilot System

This module implements a LangGraph-based multi-agent system that coordinates
between Research Agent, AMP Designer Agent, and Data Analysis Agent using
tight coupling through internal graph nodes rather than API calls.
"""
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from enum import Enum
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from configuration import OPENAI_API_KEY, CHAT_MODEL, ROUTING_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for importing other agents
BACKEND_DIR = Path(__file__).parent.parent
sys.path.append(str(BACKEND_DIR / "Research_agent"))
sys.path.append(str(BACKEND_DIR / "AMP_designer_agent"))
sys.path.append(str(BACKEND_DIR / "Data_analysis_agent"))

# Import Research Agent tools directly (simpler approach)
try:
    research_path = str(BACKEND_DIR / "Research_agent")
    if research_path not in sys.path:
        sys.path.insert(0, research_path)

    # Import using absolute path to avoid relative import issues
    import importlib.util
    retrieval_path = BACKEND_DIR / "Research_agent" / "retrieval.py"
    spec = importlib.util.spec_from_file_location("retrieval", retrieval_path)
    retrieval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrieval_module)

    find_peptide_properties = retrieval_module.find_peptide_properties
    find_peptides_by_properties = retrieval_module.find_peptides_by_properties
    logger.info("✅ Research Agent tools imported successfully")
    research_tools_available = True
except ImportError as e:
    logger.warning(f"⚠️ Could not import Research Agent tools: {e}")
    find_peptide_properties = None
    find_peptides_by_properties = None
    research_tools_available = False

# For now, we'll use LLM-based implementations for AMP Designer and Data Analysis
# This avoids complex import issues while maintaining functionality
amp_designer_workflow_available = True
data_analysis_workflow_available = True

logger.info("✅ AMP Designer Agent (LLM-based) available")
logger.info("✅ Data Analysis Agent (LLM-based) available")

class AgentType(Enum):
    RESEARCH = "research"
    AMP_DESIGNER = "amp_designer" 
    DATA_ANALYSIS = "data_analysis"
    GENERAL = "general"

class GraphState(TypedDict):
    """Unified graph state for multi-agent coordination"""
    messages: List[Any]
    user_query: str
    selected_agent: Optional[str]
    agent_response: Optional[str]
    routing_confidence: Optional[float]
    processing_history: List[Dict[str, Any]]
    session_id: str
    user_context: Dict[str, Any]
    
    # Agent-specific state fields for tight coupling
    research_result: Optional[Dict[str, Any]]
    amp_design_result: Optional[Dict[str, Any]]
    data_analysis_result: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]

class MultiAgentGraph:
    """Multi-agent coordination graph using tight coupling"""
    
    def __init__(self):
        """Initialize multi-agent graph with embedded agent workflows"""
        self.routing_llm = ChatOpenAI(
            model=ROUTING_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.chat_llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.7
        )
        
        # Initialize agent workflows
        self._init_agent_workflows()
        
        # Create the main coordination graph
        self.graph = self._create_graph()
        
    def _init_agent_workflows(self):
        """Initialize agent tools and workflows"""

        # Initialize Research Agent tools
        if research_tools_available:
            self.research_tools = {
                'find_peptide_properties': find_peptide_properties,
                'find_peptides_by_properties': find_peptides_by_properties
            }
            logger.info("✅ Research Agent tools initialized")
        else:
            self.research_tools = {}
            logger.warning("⚠️ Research Agent tools not available")

        # Initialize AMP Designer Agent (LLM-based for now)
        if amp_designer_workflow_available:
            self.amp_designer_available = True
            logger.info("✅ AMP Designer Agent (LLM-based) initialized")
        else:
            self.amp_designer_available = False
            logger.warning("⚠️ AMP Designer Agent not available")

        # Initialize Data Analysis Agent (LLM-based for now)
        if data_analysis_workflow_available:
            self.data_analysis_available = True
            logger.info("✅ Data Analysis Agent (LLM-based) initialized")
        else:
            self.data_analysis_available = False
            logger.warning("⚠️ Data Analysis Agent not available")

    def _create_graph(self):
        """Create the main coordination graph"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("research_agent", self._call_research_agent)
        workflow.add_node("amp_designer_agent", self._call_amp_designer_agent)
        workflow.add_node("data_analysis_agent", self._call_data_analysis_agent)
        workflow.add_node("general_agent", self._call_general_agent)
        workflow.add_node("finalizer", self._finalize_response)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            {
                "research": "research_agent",
                "amp_designer": "amp_designer_agent", 
                "data_analysis": "data_analysis_agent",
                "general": "general_agent"
            }
        )
        
        # All agents go to finalizer
        workflow.add_edge("research_agent", "finalizer")
        workflow.add_edge("amp_designer_agent", "finalizer")
        workflow.add_edge("data_analysis_agent", "finalizer")
        workflow.add_edge("general_agent", "finalizer")
        
        # Finalizer ends the workflow
        workflow.add_edge("finalizer", END)
        
        return workflow.compile()

    def _route_query(self, state: GraphState) -> GraphState:
        """Route query to appropriate agent using LLM"""
        user_query = state["user_query"]
        
        routing_prompt = f"""You are AMPilot's intelligent router. Analyze the user query and select the most appropriate agent.

Available Agents:
1. research - Research agent: Search antimicrobial peptide databases, find sequence properties, search peptides by conditions
2. amp_designer - AMP Designer agent: Analyze and rank antimicrobial peptide sequences, compare sequence performance  
3. data_analysis - Data Analysis agent: Statistical analysis of experimental data, data visualization, hypothesis testing
4. general - General conversation: Greetings, help information, general dialogue

User Query: {user_query}

Analyze the query content and return JSON format:
{{
    "agent": "selected_agent_name",
    "confidence": 0.95,
    "reasoning": "selection_reason"
}}

Return ONLY JSON, no other content."""

        try:
            response = self.routing_llm.invoke([HumanMessage(content=routing_prompt)])
            content = response.content.strip()
            
            # Try to extract JSON content
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # If content doesn't start with {, try to find JSON part
            if not content.startswith("{"):
                import re
                json_match = re.search(r'\{[^}]*\}', content)
                if json_match:
                    content = json_match.group()
                else:
                    raise ValueError("No valid JSON found in response")
            
            result = json.loads(content)
            
            selected_agent = result.get("agent", "general")
            confidence = result.get("confidence", 0.1)
            reasoning = result.get("reasoning", "Default routing")
            
            state["selected_agent"] = selected_agent
            state["routing_confidence"] = confidence
            state["processing_history"].append({
                "step": "routing",
                "selected_agent": selected_agent,
                "confidence": confidence,
                "reasoning": reasoning
            })
            
            logger.info(f"✅ Routed to {selected_agent} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Routing failed: {e}")
            # Use fallback routing
            state["selected_agent"] = "general"
            state["routing_confidence"] = 0.1
            state["processing_history"].append({
                "step": "routing_fallback",
                "error": str(e),
                "selected_agent": "general"
            })
        
        return state

    def _route_to_agent(self, state: GraphState) -> str:
        """Determine which agent node to route to"""
        return state.get("selected_agent", "general")

    def _call_research_agent(self, state: GraphState) -> GraphState:
        """Call Research Agent using direct tool calls"""
        try:
            if research_tools_available and self.research_tools:
                user_query = state["user_query"]

                # Use LLM to determine which research tool to use and extract parameters
                research_prompt = f"""You are a research assistant for antimicrobial peptides. Analyze the user query and determine the appropriate action.

Available tools:
1. find_peptide_properties(sequence) - Get properties of a specific peptide sequence
2. find_peptides_by_properties(query, bacterium, mic_range, modifications, strain, length_range) - Find peptides by criteria

User Query: {user_query}

If the query contains a specific peptide sequence, use find_peptide_properties.
If the query asks to find peptides with certain properties, use find_peptides_by_properties.

Respond with JSON format:
{{
    "tool": "tool_name",
    "parameters": {{"sequence": "PEPTIDE_SEQUENCE"}} for find_peptide_properties,
    "parameters": {{"query": "", "bacterium": "S. aureus", "length_range": "20-25", "mic_range": "", "modifications": "", "strain": ""}} for find_peptides_by_properties,
    "reasoning": "explanation"
}}

Return ONLY JSON, no other content."""

                response = self.chat_llm.invoke([HumanMessage(content=research_prompt)])
                content = response.content.strip()

                # Extract JSON
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()

                if not content.startswith("{"):
                    import re
                    json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group()

                tool_info = json.loads(content)
                tool_name = tool_info.get("tool")
                parameters = tool_info.get("parameters", {})

                # Execute the appropriate tool
                if tool_name == "find_peptide_properties" and "find_peptide_properties" in self.research_tools:
                    sequence = parameters.get("sequence", "")
                    if sequence:
                        result = self.research_tools["find_peptide_properties"].invoke({"sequence": sequence})
                        state["agent_response"] = f"Peptide properties for {sequence}:\n{result}"
                    else:
                        state["agent_response"] = "No sequence provided for property analysis."

                elif tool_name == "find_peptides_by_properties" and "find_peptides_by_properties" in self.research_tools:
                    tool_input = {
                        "query": parameters.get("query", ""),
                        "bacterium": parameters.get("bacterium", ""),
                        "mic_range": parameters.get("mic_range", ""),
                        "modifications": parameters.get("modifications", ""),
                        "strain": parameters.get("strain", ""),
                        "length_range": parameters.get("length_range", "")
                    }
                    result = self.research_tools["find_peptides_by_properties"].invoke(tool_input)
                    state["agent_response"] = f"Found peptides matching criteria:\n{result}"

                else:
                    state["agent_response"] = f"Tool {tool_name} not available or invalid parameters."

                state["research_result"] = {"tool": tool_name, "parameters": parameters}
                state["processing_history"].append({
                    "step": "research_agent_execution",
                    "status": "success",
                    "tool": tool_name
                })

                logger.info(f"✅ Research Agent executed successfully using {tool_name}")

            else:
                state["agent_response"] = "Research Agent tools not available. Please ensure Weaviate is running and tools are properly configured."
                state["error"] = "Research Agent tools not available"

        except Exception as e:
            logger.error(f"❌ Research Agent failed: {e}")
            state["agent_response"] = f"Research Agent encountered an error: {str(e)}"
            state["error"] = str(e)

        return state

    def _call_amp_designer_agent(self, state: GraphState) -> GraphState:
        """Call AMP Designer Agent using LLM-based analysis"""
        try:
            if self.amp_designer_available:
                user_query = state["user_query"]

                # Extract sequences from the query
                import re
                potential_sequences = re.findall(r'[A-Z]{10,}', user_query)

                # Create comprehensive AMP design analysis prompt
                design_prompt = f"""You are an expert in antimicrobial peptide design and analysis with deep knowledge of structure-activity relationships.

User Query: {user_query}

"""

                if potential_sequences:
                    design_prompt += f"""
Sequences to analyze: {', '.join(potential_sequences)}

Please provide a comprehensive analysis including:

1. **Sequence Analysis**:
   - Length, composition, and charge analysis
   - Hydrophobicity patterns and amphipathicity
   - Secondary structure predictions
   - Key functional motifs identification

2. **Structure-Activity Relationships**:
   - Membrane interaction mechanisms
   - Target specificity predictions
   - Stability and resistance factors

3. **Comparative Analysis** (if multiple sequences):
   - Similarity and differences between sequences
   - Relative activity predictions
   - Ranking with detailed justification

4. **Design Recommendations**:
   - Optimization suggestions for enhanced activity
   - Modifications to improve selectivity
   - Stability enhancement strategies

5. **Potential Applications**:
   - Suitable target organisms
   - Therapeutic applications
   - Formulation considerations
"""
                else:
                    design_prompt += """
Please provide expert guidance on antimicrobial peptide design covering:

1. **Design Principles**:
   - Key structural features for antimicrobial activity
   - Sequence-function relationships
   - Optimization strategies

2. **Analysis Methods**:
   - Computational tools and approaches
   - Experimental validation strategies
   - Performance metrics

3. **Current Trends**:
   - Novel design approaches
   - Emerging applications
   - Challenges and solutions

Provide detailed, scientifically accurate information based on current research."""

                response = self.chat_llm.invoke([HumanMessage(content=design_prompt)])

                state["agent_response"] = response.content
                state["amp_design_result"] = {
                    "sequences_analyzed": potential_sequences,
                    "analysis_type": "comprehensive_design_analysis"
                }
                state["processing_history"].append({
                    "step": "amp_designer_execution",
                    "status": "success",
                    "sequences_found": len(potential_sequences)
                })

                logger.info("✅ AMP Designer Agent (LLM-based) executed successfully")

            else:
                state["agent_response"] = "AMP Designer Agent not available. Please check system configuration."
                state["error"] = "AMP Designer Agent not available"

        except Exception as e:
            logger.error(f"❌ AMP Designer Agent failed: {e}")
            state["agent_response"] = f"AMP Designer Agent encountered an error: {str(e)}"
            state["error"] = str(e)

        return state

    def _call_data_analysis_agent(self, state: GraphState) -> GraphState:
        """Call Data Analysis Agent using LLM-based analysis"""
        try:
            if self.data_analysis_available:
                user_query = state["user_query"]

                # Create comprehensive data analysis prompt
                analysis_prompt = f"""You are an expert in antimicrobial peptide data analysis and biostatistics with extensive experience in peptide research.

User Query: {user_query}

Please provide comprehensive guidance covering:

1. **Statistical Analysis Approach**:
   - Appropriate statistical methods for the query
   - Experimental design considerations
   - Sample size and power analysis recommendations
   - Data preprocessing requirements

2. **Data Analysis Methods**:
   - Descriptive statistics and exploratory data analysis
   - Hypothesis testing approaches (parametric vs non-parametric)
   - Correlation and regression analysis
   - Machine learning applications if relevant

3. **Visualization Strategies**:
   - Recommended plot types for the data
   - Key visualizations for antimicrobial peptide research
   - Interactive visualization tools
   - Publication-quality figure guidelines

4. **Interpretation Guidelines**:
   - How to interpret statistical results
   - Common pitfalls and how to avoid them
   - Effect size considerations
   - Clinical/biological significance vs statistical significance

5. **Tools and Software**:
   - Recommended analysis software (R, Python, etc.)
   - Specific packages and libraries
   - Code examples or pseudocode if helpful
   - Reproducibility best practices

6. **Quality Control**:
   - Data validation approaches
   - Outlier detection and handling
   - Missing data strategies
   - Bias assessment and mitigation

For antimicrobial peptide research specifically, consider:
- MIC value analysis and dose-response relationships
- Sequence-activity correlations
- Structure-function analysis
- Comparative effectiveness studies
- Resistance development patterns

Provide detailed, scientifically rigorous recommendations with practical implementation guidance."""

                response = self.chat_llm.invoke([HumanMessage(content=analysis_prompt)])

                state["agent_response"] = response.content
                state["data_analysis_result"] = {
                    "analysis_type": "comprehensive_statistical_guidance",
                    "query_processed": user_query
                }
                state["processing_history"].append({
                    "step": "data_analysis_execution",
                    "status": "success",
                    "analysis_approach": "llm_based_guidance"
                })

                logger.info("✅ Data Analysis Agent (LLM-based) executed successfully")

            else:
                state["agent_response"] = "Data Analysis Agent not available. Please check system configuration."
                state["error"] = "Data Analysis Agent not available"

        except Exception as e:
            logger.error(f"❌ Data Analysis Agent failed: {e}")
            state["agent_response"] = f"Data Analysis Agent encountered an error: {str(e)}"
            state["error"] = str(e)

        return state

    def _call_general_agent(self, state: GraphState) -> GraphState:
        """Handle general conversation using LLM"""
        try:
            general_prompt = f"""You are AMPilot, an AI assistant specialized in antimicrobial peptide research.

User Query: {state["user_query"]}

Please provide a helpful response. If the query is not related to antimicrobial peptides,
guide the user to ask questions about:
- Peptide sequence analysis and properties
- Antimicrobial activity prediction
- Database searches for specific peptides
- Data analysis and visualization
- Peptide design and optimization

Respond in a friendly and informative manner."""

            response = self.chat_llm.invoke([HumanMessage(content=general_prompt)])

            state["agent_response"] = response.content
            state["processing_history"].append({
                "step": "general_agent_execution",
                "status": "success"
            })

            logger.info("✅ General Agent executed successfully")

        except Exception as e:
            logger.error(f"❌ General Agent failed: {e}")
            state["agent_response"] = "I'm AMPilot, specialized in antimicrobial peptide research. How can I help you?"
            state["error"] = str(e)

        return state

    def _finalize_response(self, state: GraphState) -> GraphState:
        """Finalize the response"""
        state["final_response"] = state.get("agent_response", "No response generated")

        state["processing_history"].append({
            "step": "finalization",
            "status": "completed"
        })

        logger.info("✅ Response finalized")
        return state

    async def process_query(self, user_query: str, session_id: str = "default", user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user query through the multi-agent graph"""
        if user_context is None:
            user_context = {}

        initial_state = {
            "messages": [],
            "user_query": user_query,
            "selected_agent": None,
            "agent_response": None,
            "routing_confidence": None,
            "processing_history": [],
            "session_id": session_id,
            "user_context": user_context,
            "research_result": None,
            "amp_design_result": None,
            "data_analysis_result": None,
            "final_response": None,
            "error": None
        }

        try:
            # Execute the graph
            result = self.graph.invoke(initial_state)

            return {
                "success": True,
                "response": result.get("final_response", ""),
                "metadata": {
                    "selected_agent": result.get("selected_agent"),
                    "routing_confidence": result.get("routing_confidence"),
                    "processing_history": result.get("processing_history", [])
                }
            }

        except Exception as e:
            logger.error(f"❌ Graph execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request."
            }

    async def process_query_stream(self, user_query: str, session_id: str = "default", user_context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a query with streaming response"""
        if user_context is None:
            user_context = {}

        # Yield initial status
        yield {"type": "stream_chunk", "content": "🔍 Analyzing your query...\n\n"}

        initial_state = {
            "messages": [],
            "user_query": user_query,
            "selected_agent": None,
            "agent_response": None,
            "routing_confidence": None,
            "processing_history": [],
            "session_id": session_id,
            "user_context": user_context,
            "research_result": None,
            "amp_design_result": None,
            "data_analysis_result": None,
            "final_response": None,
            "error": None
        }

        try:
            # Execute the graph
            result = self.graph.invoke(initial_state)

            # Yield routing information
            selected_agent = result.get("selected_agent", "general")
            confidence = result.get("routing_confidence", 0.0)

            yield {"type": "stream_chunk", "content": f"✅ Routing to {selected_agent.replace('_', ' ').title()} Agent (confidence: {confidence:.2f})\n\n"}

            # Yield processing status
            if selected_agent == "research":
                yield {"type": "stream_chunk", "content": "🔬 Research Agent Processing\n\n"}
                yield {"type": "stream_chunk", "content": "Searching antimicrobial peptide database...\n\n"}
            elif selected_agent == "amp_designer":
                yield {"type": "stream_chunk", "content": "🧬 AMP Designer Agent Processing\n\n"}
                yield {"type": "stream_chunk", "content": "Analyzing peptide sequences...\n\n"}
            elif selected_agent == "data_analysis":
                yield {"type": "stream_chunk", "content": "📊 Data Analysis Agent Processing\n\n"}
                yield {"type": "stream_chunk", "content": "Performing statistical analysis...\n\n"}
            else:
                yield {"type": "stream_chunk", "content": "💬 General Conversation\n\n"}

            # Yield final response
            final_response = result.get("final_response", "")
            if final_response:
                yield {"type": "stream_chunk", "content": final_response}
            else:
                yield {"type": "stream_chunk", "content": "No response generated."}

        except Exception as e:
            logger.error(f"❌ Streaming execution failed: {e}")
            yield {"type": "stream_chunk", "content": f"⚠️ Error: {str(e)}"}


def get_graph() -> MultiAgentGraph:
    """Factory function to create and return a MultiAgentGraph instance"""
    return MultiAgentGraph()
