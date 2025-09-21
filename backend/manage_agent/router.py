"""
Intelligent Agent Router for AMPilot Multi-Agent System

This module implements intelligent routing logic to direct user queries
to the appropriate specialized agent based on content analysis and intent detection.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from configuration import (
    OPENAI_API_KEY, ROUTING_MODEL, AGENT_CAPABILITIES,
    ROUTER_REQUIRE_LLM, ROUTER_DISABLE_KEYWORDS
)
from prompts import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enumeration of available agent types"""
    RESEARCH = "research"
    AMP_DESIGNER = "amp_designer" 
    DATA_ANALYSIS = "data_analysis"
    UNKNOWN = "unknown"

@dataclass
class RoutingResult:
    """Result of agent routing decision"""
    agent_type: AgentType
    confidence: float
    reasoning: str
    extracted_info: Dict[str, Any]
    fallback_agents: List[AgentType]

class AgentRouter:
    """
    Intelligent router for directing queries to appropriate agents
    """
    
    def __init__(self):
        """Initialize the agent router"""
        self.llm = ChatOpenAI(
            model=ROUTING_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        ) if OPENAI_API_KEY else None
        
        logger.info("Agent Router initialized")
    
    async def route_query(self, user_query: str, context: Optional[Dict] = None) -> RoutingResult:
        """
        Route user query to the most appropriate agent using LLM-based intelligent routing

        Args:
            user_query: User's input query
            context: Optional context information (file uploads, session history, etc.)

        Returns:
            RoutingResult with agent selection and reasoning
        """
        try:
            # Try LLM-based routing first
            if self.llm:
                try:
                    result = await self._llm_based_routing(user_query, context)
                    logger.info(f"LLM routing: {result.agent_type.value} - {result.reasoning}")
                    return result
                except Exception as e:
                    if ROUTER_REQUIRE_LLM:
                        logger.error(f"LLM routing failed and ROUTER_REQUIRE_LLM=True: {e}")
                        return self._fallback_routing(user_query, context)
                    else:
                        logger.warning(f"LLM routing failed, using intelligent fallback: {e}")
                        return self._intelligent_fallback_routing(user_query, context)
            else:
                if ROUTER_REQUIRE_LLM:
                    logger.error("LLM not available and ROUTER_REQUIRE_LLM=True; returning UNKNOWN")
                    return self._fallback_routing(user_query, context)
                else:
                    logger.warning("LLM not available, using intelligent fallback routing")
                    return self._intelligent_fallback_routing(user_query, context)

        except Exception as e:
            logger.error(f"Routing failed unexpectedly: {e}")
            return self._fallback_routing(user_query, context)


    async def _llm_based_routing(self, user_query: str, context: Optional[Dict] = None) -> RoutingResult:
        """
        LLM-based routing for complex queries

        Args:
            user_query: User's input query
            context: Optional context information

        Returns:
            RoutingResult from LLM analysis
        """
        try:
            # Prepare context information
            context_str = ""
            if context:
                if context.get("has_file_upload"):
                    context_str += f"User has uploaded a file: {context.get('file_info', {})}\n"
                if context.get("session_history"):
                    context_str += f"Recent conversation context available\n"

            # System prompt for routing
            system_prompt = ROUTER_SYSTEM_PROMPT

            user_message = f"""Context: {context_str}

User Query: {user_query}

Route this query to the most appropriate agent."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]

            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
                
                agent_type = AgentType(result.get("agent", "unknown"))
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "LLM-based routing")
                extracted_info = result.get("extracted_info", {})
                
                # Determine fallback agents
                all_agents = [AgentType.RESEARCH, AgentType.AMP_DESIGNER, AgentType.DATA_ANALYSIS]
                fallback_agents = [a for a in all_agents if a != agent_type]
                
                return RoutingResult(
                    agent_type=agent_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    extracted_info=extracted_info,
                    fallback_agents=fallback_agents
                )
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse LLM routing response: {e}")
                # Fallback to research agent with low confidence
                return RoutingResult(
                    agent_type=AgentType.RESEARCH,
                    confidence=0.3,
                    reasoning="LLM response parsing failed, defaulting to research agent",
                    extracted_info={},
                    fallback_agents=[AgentType.AMP_DESIGNER, AgentType.DATA_ANALYSIS]
                )
                
        except Exception as e:
            logger.error(f"LLM-based routing failed: {e}")
            return RoutingResult(
                agent_type=AgentType.RESEARCH,
                confidence=0.2,
                reasoning=f"LLM routing error: {str(e)}",
                extracted_info={},
                fallback_agents=[AgentType.AMP_DESIGNER, AgentType.DATA_ANALYSIS]
            )
    

    def get_agent_info(self, agent_type: AgentType) -> Dict[str, Any]:
        """
        Get information about a specific agent
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Agent information dictionary
        """
        if agent_type == AgentType.UNKNOWN:
            return {"error": "Unknown agent type"}

        return AGENT_CAPABILITIES.get(agent_type.value, {})

    def _intelligent_fallback_routing(self, user_query: str, context: Optional[Dict] = None) -> RoutingResult:
        """
        Keyword-based fallback routing (disabled when ROUTER_DISABLE_KEYWORDS=True)
        """
        if ROUTER_DISABLE_KEYWORDS:
            # Return UNKNOWN to force LLM usage only
            return RoutingResult(
                agent_type=AgentType.UNKNOWN,
                confidence=0.1,
                reasoning="Keyword fallback disabled; LLM routing required",
                extracted_info={},
                fallback_agents=[AgentType.RESEARCH, AgentType.AMP_DESIGNER, AgentType.DATA_ANALYSIS]
            )

        # (Legacy keyword heuristic kept for optional use if explicitly enabled)
        query_lower = user_query.lower().strip()

        # Extract sequences (uppercase amino acid patterns)
        import re
        sequences = re.findall(r'\b[ACDEFGHIKLMNPQRSTVWY]{8,}\b', user_query.upper())

        # Heuristic: multiple sequences → AMP Designer
        if len(sequences) >= 2:
            return RoutingResult(
                agent_type=AgentType.AMP_DESIGNER,
                confidence=0.8,
                reasoning=f"Multiple sequences detected ({len(sequences)}) - routing to AMP Designer",
                extracted_info={"sequences": sequences},
                fallback_agents=[AgentType.RESEARCH, AgentType.DATA_ANALYSIS]
            )

        # Default to research with low confidence if heuristics are minimal
        return RoutingResult(
            agent_type=AgentType.RESEARCH,
            confidence=0.3,
            reasoning="Minimal heuristic fallback used",
            extracted_info={"sequences": sequences} if sequences else {},
            fallback_agents=[AgentType.AMP_DESIGNER, AgentType.DATA_ANALYSIS]
        )

    def _fallback_routing(self, user_query: str, context: Optional[Dict] = None) -> RoutingResult:
        """Strict fallback when LLM is required or routing fails"""
        return RoutingResult(
            agent_type=AgentType.UNKNOWN,
            confidence=0.05,
            reasoning="LLM routing unavailable or failed; strict fallback to UNKNOWN",
            extracted_info={},
            fallback_agents=[AgentType.RESEARCH, AgentType.AMP_DESIGNER, AgentType.DATA_ANALYSIS]
        )


