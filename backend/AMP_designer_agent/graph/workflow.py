"""
AMP Designer Agent Workflow

This module implements the antimicrobial peptide analysis workflow using LangGraph,
providing a structured approach to sequence analysis, experience learning, and ranking.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
import logging
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from configuration import OPENAI_API_KEY, CHAT_MODEL
from graph.experience_manager import ExperienceManager
from tools.sequence_embedder import SequenceEmbedder

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State structure for the LangGraph workflow"""
    # Input data
    sequence_pair: Optional[List[str]]
    sequences_to_rank: Optional[List[str]]
    user_requirements: Optional[Dict[str, Any]]
    
    # Processing state
    current_step: str
    embeddings: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    experience_insights: Optional[List[str]]
    ranking_results: Optional[List[Dict[str, Any]]]
    
    # Output
    final_output: Optional[Dict[str, Any]]
    error_message: Optional[str]

@dataclass
class AMPAnalysisWorkflow:
    """LangGraph-based workflow for AMP sequence analysis"""
    
    def __init__(self):
        """Initialize the workflow with required components"""
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        self.experience_manager = ExperienceManager()
        self.embedder = SequenceEmbedder()
        self.graph = self._build_graph()
        
        logger.info("AMPAnalysisWorkflow initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("generate_embeddings", self._generate_embeddings)
        workflow.add_node("analyze_sequences", self._analyze_sequences)
        workflow.add_node("store_experience", self._store_experience)
        workflow.add_node("rank_sequences", self._rank_sequences)
        workflow.add_node("format_output", self._format_output)
        
        # Define the flow
        workflow.set_entry_point("validate_input")
        
        # Conditional routing based on task type
        workflow.add_conditional_edges(
            "validate_input",
            self._route_task,
            {
                "sequence_analysis": "generate_embeddings",
                "sequence_ranking": "rank_sequences",
                "error": END
            }
        )
        
        # Sequence analysis flow
        workflow.add_edge("generate_embeddings", "analyze_sequences")
        workflow.add_edge("analyze_sequences", "store_experience")
        workflow.add_edge("store_experience", "format_output")
        
        # Sequence ranking flow
        workflow.add_edge("rank_sequences", "format_output")
        
        # End flow
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
    
    def _validate_input(self, state: AgentState) -> AgentState:
        """Validate input data and determine task type"""
        try:
            if state.get("sequence_pair"):
                if len(state["sequence_pair"]) != 2:
                    state["error_message"] = "Sequence pair must contain exactly 2 sequences"
                    return state
                state["current_step"] = "sequence_analysis"
                
            elif state.get("sequences_to_rank"):
                if len(state["sequences_to_rank"]) < 2:
                    state["error_message"] = "Must provide at least 2 sequences to rank"
                    return state
                state["current_step"] = "sequence_ranking"
                
            else:
                state["error_message"] = "Must provide either sequence_pair or sequences_to_rank"
                return state
                
            logger.info(f"Input validated for task: {state['current_step']}")
            return state
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            state["error_message"] = f"Input validation failed: {str(e)}"
            return state
    
    def _route_task(self, state: AgentState) -> str:
        """Route to appropriate task based on input"""
        if state.get("error_message"):
            return "error"
        elif state.get("current_step") == "sequence_analysis":
            return "sequence_analysis"
        elif state.get("current_step") == "sequence_ranking":
            return "sequence_ranking"
        else:
            return "error"
    
    def _generate_embeddings(self, state: AgentState) -> AgentState:
        """Generate embeddings for sequence pair"""
        try:
            sequence_pair = state["sequence_pair"]
            embeddings = {}
            
            for i, seq in enumerate(sequence_pair):
                embedding = self.embedder.encode_sequence(seq)
                embeddings[f"sequence_{i+1}"] = embedding.tolist()
            
            state["embeddings"] = embeddings
            logger.info("Generated embeddings for sequence pair")
            return state
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            state["error_message"] = f"Failed to generate embeddings: {str(e)}"
            return state
    
    def _analyze_sequences(self, state: AgentState) -> AgentState:
        """Analyze sequence pair using LLM with experience context"""
        try:
            sequence_pair = state["sequence_pair"]
            
            # Get relevant experiences
            relevant_experiences = self.experience_manager.get_relevant_experiences(
                str(sequence_pair), top_k=5
            )
            
            # Build context from experiences
            experience_context = ""
            if relevant_experiences:
                experience_context = "\\n\\nRelevant past insights:\\n"
                for exp in relevant_experiences:
                    for insight in exp.insights:
                        experience_context += f"- {insight}\\n"
            
            # Get system prompt from prompts module
            try:
                from prompts import EXPERIENCE_LEARNING_PROMPT_TEMPLATE
                # Extract just the system part before the analysis context
                system_prompt = EXPERIENCE_LEARNING_PROMPT_TEMPLATE.split("## Analysis Context:")[0].strip()
            except ImportError:
                system_prompt = """You are an expert antimicrobial peptide (AMP) researcher specializing in structure-function relationships. Analyze the provided sequence pair and extract transferable insights.

CRITICAL REQUIREMENTS - ABSOLUTE PROHIBITIONS:
1. NEVER use relative references like "sequence 1", "sequence 2", "first sequence", "second sequence", "peptide A", "peptide B", "one sequence", "the other sequence", "compared to", "higher than", "lower than"
2. NEVER make comparative statements between the two sequences
3. NEVER use phrases like "exhibits higher/lower", "shows greater/lesser", "has more/fewer"
4. Each insight must be INDEPENDENTLY VALID without knowing which specific sequences were analyzed
5. Focus on GENERAL PRINCIPLES that apply to antimicrobial peptides broadly

ANALYSIS REQUIREMENTS:
- Analyze physicochemical properties (charge, hydrophobicity, amphipathicity)
- Consider structural features (secondary structure, membrane interaction)
- Evaluate antimicrobial mechanisms (membrane disruption, intracellular targets)
- Assess sequence-activity relationships

OUTPUT FORMAT:
Provide exactly 4 scientific insights as a JSON array. Each insight should:
- Be a standalone scientific principle
- Focus on general AMP characteristics
- Avoid any sequence-specific references
- Be applicable to future AMP analysis

Example format:
["Insight 1 about general AMP properties", "Insight 2 about mechanisms", "Insight 3 about structure", "Insight 4 about activity"]"""

            # Prepare sequence data
            seq1, seq2 = sequence_pair
            user_message = f"""Analyze these antimicrobial peptide sequences:

Sequence A: {seq1}
Sequence B: {seq2}

{experience_context}

Provide 4 transferable scientific insights about antimicrobial peptide properties and mechanisms."""

            # Get LLM analysis
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse insights
            try:
                insights = json.loads(response.content)
                if not isinstance(insights, list) or len(insights) != 4:
                    raise ValueError("Expected exactly 4 insights")
            except:
                # Fallback parsing
                insights = [line.strip('- ').strip() for line in response.content.split('\\n') 
                          if line.strip() and not line.startswith('[') and not line.startswith(']')]
                insights = insights[:4]  # Take first 4
            
            # Filter insights to remove problematic ones
            filtered_insights = self._filter_insights(insights)
            
            state["analysis_results"] = {
                "insights": filtered_insights,
                "raw_response": response.content,
                "experience_context_used": len(relevant_experiences)
            }
            
            logger.info(f"Generated {len(filtered_insights)} filtered insights")
            return state
            
        except Exception as e:
            logger.error(f"Sequence analysis error: {e}")
            state["error_message"] = f"Failed to analyze sequences: {str(e)}"
            return state
    
    def _filter_insights(self, insights: List[str]) -> List[str]:
        """Filter out insights with sequence-specific references or comparisons"""
        # Expanded keywords to avoid in insights - any comparative or sequence-specific references
        avoid_keywords = [
            # Direct sequence references (with and without spaces)
            "sequence 1", "sequence 2", "sequence a", "sequence b", "seq1", "seq2",
            "sequence1", "sequence2", "sequencea", "sequenceb",  # Added variants without spaces
            "first sequence", "second sequence", "peptide 1", "peptide 2",
            "peptide a", "peptide b", "first peptide", "second peptide",
            "one sequence", "other sequence", "one peptide", "other peptide",
            "the first", "the second", "the other", "this sequence", "that sequence",

            # Comparative phrases with sequence references
            "sequence 1 exhibits", "sequence 2 exhibits", "sequence1 exhibits", "sequence2 exhibits",
            "sequence 1 shows", "sequence 2 shows", "sequence1 shows", "sequence2 shows",
            "sequence 1 has", "sequence 2 has", "sequence1 has", "sequence2 has",
            "higher lipophilicity in sequence", "lower lipophilicity in sequence",
            "broader activity compared", "narrower activity compared",
            
            # General comparative phrases
            "compared to", "in contrast to", "while the other", "whereas the other",
            "higher than", "lower than", "greater than", "less than", "more than",
            "exhibits higher", "exhibits lower", "shows greater", "shows lesser",
            "has more", "has fewer", "has higher", "has lower", "displays greater",
            "displays lower", "demonstrates higher", "demonstrates lower",
            "broader spectrum of activity against", "wider spectrum of activity against"
        ]
        
        filtered = []
        for insight in insights:
            insight_lower = insight.lower()
            
            # Check if insight contains any problematic keywords
            contains_problematic = any(keyword in insight_lower for keyword in avoid_keywords)
            
            if not contains_problematic and len(insight.strip()) > 10:
                filtered.append(insight.strip())
        
        # Ensure we have at least 2 insights, take the best ones
        if len(filtered) >= 2:
            return filtered[:2]  # Take top 2 filtered insights
        elif len(filtered) == 1:
            return filtered
        else:
            # If no insights pass filter, return generic fallback
            return ["Cationic residues enhance membrane binding affinity in antimicrobial peptides.",
                   "Amphipathic structures facilitate membrane disruption mechanisms."]

    def _store_experience(self, state: AgentState) -> AgentState:
        """Store the analysis as experience for future use"""
        try:
            sequence_pair = state["sequence_pair"]
            analysis_results = state["analysis_results"]

            # Store experience
            experience_id = self.experience_manager.add_experience(
                content=f"Analysis of sequence pair: {sequence_pair[0][:10]}... vs {sequence_pair[1][:10]}...",
                context="Sequence length comparison and amino acid composition analysis",
                sequence_pair=sequence_pair,
                insights=analysis_results["insights"],
                confidence=0.85
            )

            # Save experiences to file
            self.experience_manager.save_experiences()

            state["experience_insights"] = analysis_results["insights"]
            logger.info(f"Stored experience with ID: {experience_id}")
            return state

        except Exception as e:
            logger.error(f"Experience storage error: {e}")
            state["error_message"] = f"Failed to store experience: {str(e)}"
            return state

    def _rank_sequences(self, state: AgentState) -> AgentState:
        """Rank sequences based on user requirements"""
        try:
            sequences = state["sequences_to_rank"]
            requirements = state.get("user_requirements", {})

            # Simple ranking based on sequence length and charge for now
            # This can be enhanced with more sophisticated ranking logic
            ranked_sequences = []

            for i, seq in enumerate(sequences):
                # Calculate basic properties
                length = len(seq)
                charge = sum(1 for aa in seq if aa in 'KRH') - sum(1 for aa in seq if aa in 'DE')
                hydrophobic_ratio = sum(1 for aa in seq if aa in 'AILMFPWV') / length

                score = length * 0.3 + charge * 0.4 + hydrophobic_ratio * 0.3

                ranked_sequences.append({
                    "sequence": seq,
                    "rank": i + 1,
                    "score": score,
                    "properties": {
                        "length": length,
                        "charge": charge,
                        "hydrophobic_ratio": hydrophobic_ratio
                    }
                })

            # Sort by score (descending)
            ranked_sequences.sort(key=lambda x: x["score"], reverse=True)

            # Update ranks
            for i, item in enumerate(ranked_sequences):
                item["rank"] = i + 1

            state["ranking_results"] = ranked_sequences
            logger.info(f"Ranked {len(sequences)} sequences")
            return state

        except Exception as e:
            logger.error(f"Sequence ranking error: {e}")
            state["error_message"] = f"Failed to rank sequences: {str(e)}"
            return state

    def _format_output(self, state: AgentState) -> AgentState:
        """Format the final output"""
        try:
            if state.get("error_message"):
                state["final_output"] = {
                    "success": False,
                    "error": state["error_message"]
                }
            elif state.get("current_step") == "sequence_analysis":
                state["final_output"] = {
                    "success": True,
                    "task_type": "sequence_analysis",
                    "insights": state.get("experience_insights", []),
                    "analysis_details": state.get("analysis_results", {})
                }
            elif state.get("current_step") == "sequence_ranking":
                state["final_output"] = {
                    "success": True,
                    "task_type": "sequence_ranking",
                    "ranked_sequences": state.get("ranking_results", [])
                }
            else:
                state["final_output"] = {
                    "success": False,
                    "error": "Unknown task type"
                }

            logger.info("Formatted final output")
            return state

        except Exception as e:
            logger.error(f"Output formatting error: {e}")
            state["final_output"] = {
                "success": False,
                "error": f"Failed to format output: {str(e)}"
            }
            return state

    def analyze_sequence_pair(self, sequence1: str, sequence2: str) -> Dict[str, Any]:
        """Analyze a pair of sequences"""
        initial_state = AgentState(
            sequence_pair=[sequence1, sequence2],
            sequences_to_rank=None,
            user_requirements=None,
            current_step="",
            embeddings=None,
            analysis_results=None,
            experience_insights=None,
            ranking_results=None,
            final_output=None,
            error_message=None
        )

        result = self.graph.invoke(initial_state)
        return result["final_output"]

    def rank_sequences(self, sequences: List[str], requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """Rank a list of sequences"""
        initial_state = AgentState(
            sequence_pair=None,
            sequences_to_rank=sequences,
            user_requirements=requirements or {},
            current_step="",
            embeddings=None,
            analysis_results=None,
            experience_insights=None,
            ranking_results=None,
            final_output=None,
            error_message=None
        )

        result = self.graph.invoke(initial_state)
        return result["final_output"]
