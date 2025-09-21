"""
Clean Multi-Agent Workflow using LangGraph

This module implements a simplified multi-agent workflow that orchestrates
the three specialized agents (Research, AMP Designer, Data Analysis)
using actual HTTP calls to their APIs, without any hardcoded responses.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from configuration import (
    OPENAI_API_KEY, CHAT_MODEL,
    RESEARCH_AGENT_URL, AMP_DESIGNER_AGENT_URL, DATA_ANALYSIS_AGENT_URL
)
from router import AgentRouter, AgentType, RoutingResult

logger = logging.getLogger(__name__)

class MultiAgentState(TypedDict):
    """State structure for the multi-agent workflow"""
    # Input
    user_query: str
    user_context: Optional[Dict[str, Any]]
    session_id: str
    
    # Routing
    routing_result: Optional[RoutingResult]
    selected_agent: Optional[str]
    
    # Agent Communication
    agent_request: Optional[Dict[str, Any]]
    agent_response: Optional[Dict[str, Any]]
    
    # Output
    final_response: Optional[str]
    metadata: Optional[Dict[str, Any]]
    processing_history: List[Dict[str, Any]]

class MultiAgentWorkflow:
    """Clean multi-agent workflow implementation"""
    
    def __init__(self):
        """Initialize the workflow"""
        self.router = AgentRouter()
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        self.graph = self._build_graph()
        logger.info("Multi-Agent Workflow initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("call_agent", self._call_agent)
        workflow.add_node("process_response", self._process_response)
        
        # Add edges
        workflow.set_entry_point("route_query")
        workflow.add_edge("route_query", "call_agent")
        workflow.add_edge("call_agent", "process_response")
        workflow.add_edge("process_response", END)
        
        return workflow.compile()
    
    async def _route_query(self, state: MultiAgentState) -> MultiAgentState:
        """Route the user query to appropriate agent"""
        try:
            routing_result = await self.router.route_query(state["user_query"])
            
            state["routing_result"] = routing_result
            state["selected_agent"] = routing_result.agent_type.value
            state["processing_history"].append({
                "step": "routing",
                "timestamp": datetime.now().isoformat(),
                "agent_selected": routing_result.agent_type.value,
                "confidence": routing_result.confidence,
                "reasoning": routing_result.reasoning
            })
            
            logger.info(f"Query routed to {routing_result.agent_type.value} with confidence {routing_result.confidence}")
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            state["routing_result"] = None
            state["selected_agent"] = "research"  # fallback
            
        return state
    
    async def _call_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Call the selected agent via HTTP"""
        try:
            agent_type = state["routing_result"].agent_type if state["routing_result"] else AgentType.RESEARCH

            # Handle UNKNOWN agent type with a default response
            if agent_type == AgentType.UNKNOWN:
                state["agent_response"] = {
                    "success": True,
                    "response": """Hello! I'm AMPilot, your antimicrobial peptide research assistant.

I can help you with:
🔬 **Research**: Find peptides by properties, sequences, or target organisms
🧬 **Design**: Analyze and rank peptide sequences for activity
📊 **Analysis**: Perform statistical analysis on peptide data

**Example queries:**
- "Find antimicrobial peptides effective against E. coli"
- "Rank these sequences by antimicrobial activity"
- "Analyze the correlation between peptide charge and MIC values"

How can I assist you with your antimicrobial peptide research?""",
                    "metadata": {"agent": "system", "type": "greeting"},
                    "agent": "system"
                }

                state["processing_history"].append({
                    "step": "agent_call",
                    "timestamp": datetime.now().isoformat(),
                    "agent": "system",
                    "success": True
                })

                return state

            # Prepare request data
            request_data = {
                "message": state["user_query"],
                "session_id": state["session_id"]
            }

            # Get agent URL
            agent_urls = {
                AgentType.RESEARCH: RESEARCH_AGENT_URL,
                AgentType.AMP_DESIGNER: AMP_DESIGNER_AGENT_URL,
                AgentType.DATA_ANALYSIS: DATA_ANALYSIS_AGENT_URL
            }

            url = f"{agent_urls[agent_type]}/chat"
            
            # Make HTTP call
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for comprehensive analysis
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=request_data) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        state["agent_response"] = {
                            "success": True,
                            "response": data.get("response", ""),
                            "metadata": data.get("metadata", {}),
                            "agent": agent_type.value
                        }
                    else:
                        error_text = await resp.text()
                        state["agent_response"] = {
                            "success": False,
                            "error": f"HTTP {resp.status}: {error_text}",
                            "agent": agent_type.value
                        }
            
            state["processing_history"].append({
                "step": "agent_call",
                "timestamp": datetime.now().isoformat(),
                "agent": agent_type.value,
                "success": state["agent_response"]["success"]
            })
            
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error occurred during agent call: {type(e).__name__}"
            logger.error(f"Agent call failed: {error_msg}")
            state["agent_response"] = {
                "success": False,
                "error": error_msg,
                "agent": state.get("selected_agent", "unknown")
            }
            
        return state
    
    async def _process_response(self, state: MultiAgentState) -> MultiAgentState:
        """Process and finalize the agent response"""
        try:
            if state["agent_response"]["success"]:
                state["final_response"] = state["agent_response"]["response"]
            else:
                state["final_response"] = f"Sorry, there was an error processing your request: {state['agent_response']['error']}"
            
            # Build metadata
            state["metadata"] = {
                "agent_used": state.get("selected_agent", "unknown"),
                "routing_confidence": state["routing_result"].confidence if state["routing_result"] else 0.0,
                "processing_time": datetime.now().isoformat(),
                "total_processing_steps": len(state["processing_history"]),
                "session_id": state["session_id"]
            }
            
            state["processing_history"].append({
                "step": "response_processing",
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Response processing failed: {e}")
            state["final_response"] = f"Sorry, an error occurred while processing the response: {str(e)}"
            
        return state
    
    async def process_query(self, user_query: str, session_id: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single query through the workflow"""
        try:
            # Initialize state
            initial_state = MultiAgentState(
                user_query=user_query,
                user_context=user_context or {},
                session_id=session_id,
                routing_result=None,
                selected_agent=None,
                agent_request=None,
                agent_response=None,
                final_response=None,
                metadata=None,
                processing_history=[]
            )
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Return structured response
            return {
                "response": final_state["final_response"],
                "metadata": final_state["metadata"],
                "processing_history": final_state["processing_history"],
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "response": f"Sorry, an error occurred: {str(e)}",
                "metadata": {"error": str(e)},
                "processing_history": [],
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "success": False
            }
    
    async def process_query_stream(self, user_query: str, session_id: str, user_context: Optional[Dict[str, Any]] = None):
        """Process query with streaming support"""
        try:
            # Process the query normally
            result = await self.process_query(user_query, session_id, user_context)

            # Stream the response content in chunks
            if result.get("success") and result.get("response"):
                response_text = result["response"]

                # Split response into chunks for streaming effect
                chunk_size = 50  # characters per chunk
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    yield {
                        "type": "stream_chunk",
                        "content": chunk
                    }
                    # Small delay for streaming effect
                    await asyncio.sleep(0.05)
            else:
                # Handle error case
                error_msg = result.get("response", "Unknown error occurred")
                yield {
                    "type": "stream_chunk",
                    "content": error_msg
                }

        except Exception as e:
            logger.error(f"Streaming query processing failed: {e}")
            yield {
                "type": "stream_chunk",
                "content": f"Sorry, an error occurred: {str(e)}"
            }
