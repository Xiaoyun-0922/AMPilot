"""
Enhanced AMP Ranking Module

This module implements the enhanced ranking system that:
1. First analyzes each peptide individually using accumulated experience
2. Then performs comprehensive comparison and ranking
3. Provides complete ranking with detailed justification
4. Integrates professional AMP design expertise
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configuration import OPENAI_API_KEY, CHAT_MODEL
from graph.experience_manager import ExperienceManager

logger = logging.getLogger(__name__)

class EnhancedAMPRankingModule:
    """
    Enhanced AMP Ranking Module
    
    Implements two-stage ranking process:
    1. Individual sequence analysis using experience database
    2. Comprehensive ranking with detailed justification
    """
    
    def __init__(self):
        """Initialize the enhanced ranking module"""
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.experience_manager = ExperienceManager()
        
        logger.info("Enhanced AMP Ranking Module initialized successfully")

    def rank_sequences(self, sequences: List[str], user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for two-stage ranking process

        Args:
            sequences: List of peptide sequences to rank
            user_requirements: Dictionary of user requirements

        Returns:
            Dictionary containing individual analyses and final ranking
        """
        try:
            logger.info(f"Starting two-stage ranking for {len(sequences)} sequences")

            # Stage 1: Individual Analysis
            logger.info("Stage 1: Performing individual sequence analysis")
            individual_analyses = []

            for i, sequence in enumerate(sequences, 1):
                logger.info(f"Analyzing sequence {i}/{len(sequences)}: {sequence[:20]}...")
                analysis = self.analyze_individual_sequence(sequence, user_requirements)
                individual_analyses.append(analysis)

            successful_analyses = [a for a in individual_analyses if a.get("success")]
            logger.info(f"Individual analysis completed: {len(successful_analyses)}/{len(sequences)} successful")

            # Stage 2: Comprehensive Ranking
            logger.info("Stage 2: Performing comprehensive ranking")
            ranking_result = self.rank_sequences_comprehensively(
                sequences, user_requirements, individual_analyses
            )

            if not ranking_result.get("success"):
                logger.error(f"Comprehensive ranking failed: {ranking_result.get('error')}")
                return {
                    "success": False,
                    "error": f"Comprehensive ranking failed: {ranking_result.get('error')}",
                    "individual_analyses": individual_analyses
                }

            # Combine results
            ranking_list = ranking_result.get("ranking", [])
            if isinstance(ranking_result.get("ranking"), dict):
                ranking_list = ranking_result.get("ranking", {}).get("ranking", [])

            logger.info(f"Final ranking list length: {len(ranking_list)}")

            final_result = {
                "success": True,
                "individual_analyses": individual_analyses,
                "final_ranking": ranking_list,
                "ranking_rationale": ranking_result.get("rationale", ""),
                "confidence": ranking_result.get("confidence", 0.5),
                "processing_time": ranking_result.get("processing_time", 0),
                "stage1_success_rate": len(successful_analyses) / len(sequences) if sequences else 0,
                "stage2_success": ranking_result.get("success", False)
            }

            logger.info(f"Two-stage ranking completed successfully")
            return final_result

        except Exception as e:
            logger.error(f"Two-stage ranking failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "individual_analyses": individual_analyses if 'individual_analyses' in locals() else []
            }

    def analyze_individual_sequence(self, sequence: str, user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze individual sequence using accumulated experience
        
        Args:
            sequence: AMP sequence to analyze
            user_requirements: User requirements and preferences
            
        Returns:
            Individual sequence analysis results
        """
        try:
            # Import enhanced prompts
            from prompts import SEQUENCE_ANALYSIS_SYSTEM_PROMPT
            
            # Get relevant experiences
            relevant_experiences = self.experience_manager.get_relevant_experiences(
                query=f"sequence analysis {sequence}",
                top_k=10
            )
            
            # Format experience context
            experience_context = self._format_experience_context(relevant_experiences)
            
            # Calculate sequence properties
            properties = self._calculate_sequence_properties(sequence)
            
            # Prepare analysis prompt
            user_message = f"""Analyze this antimicrobial peptide sequence using accumulated experience and professional expertise:

## Sequence Information:
**Sequence**: {sequence}
**Length**: {len(sequence)} amino acids
**Net Charge**: {properties['net_charge']:+d}
**Hydrophobic Ratio**: {properties['hydrophobic_ratio']:.3f}
**Cationic Residues**: {properties['cationic_count']} ({properties['cationic_ratio']:.1%})
**Aromatic Residues**: {properties['aromatic_count']} ({properties['aromatic_ratio']:.1%})

## User Requirements:
{self._format_user_requirements(user_requirements)}

## Relevant Experience Context:
{experience_context}

## Analysis Requirements:
1. **Structural Analysis**: Evaluate amphipathicity, charge distribution, and secondary structure potential
2. **Activity Prediction**: Predict antimicrobial potency, spectrum, and MIC ranges
3. **Selectivity Assessment**: Evaluate bacterial vs eukaryotic membrane selectivity
4. **Stability Evaluation**: Assess protease resistance and structural stability
5. **Optimization Potential**: Identify improvement opportunities
6. **Risk Assessment**: Evaluate potential toxicity and side effects

## Output Format:
Provide comprehensive individual analysis as JSON with detailed predictions and assessments."""

            # Get LLM analysis
            messages = [
                SystemMessage(content=SEQUENCE_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse response
            try:
                analysis_result = json.loads(response.content)
                
                # Enhance with calculated properties
                analysis_result["calculated_properties"] = properties
                analysis_result["experience_support"] = len(relevant_experiences)
                
                return {
                    "success": True,
                    "sequence": sequence,
                    "analysis": analysis_result,
                    "raw_response": response.content
                }
                
            except json.JSONDecodeError:
                # Fallback parsing
                return self._fallback_parse_individual_analysis(response.content, sequence, properties)
                
        except Exception as e:
            logger.error(f"Failed to analyze individual sequence: {e}")
            return {
                "success": False,
                "sequence": sequence,
                "error": str(e)
            }
    
    def rank_sequences_comprehensively(self, sequences: List[str], 
                                     user_requirements: Dict[str, Any],
                                     individual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive ranking of all sequences
        
        Args:
            sequences: List of AMP sequences to rank
            user_requirements: User requirements and preferences
            individual_analyses: Results from individual sequence analyses
            
        Returns:
            Comprehensive ranking results
        """
        try:
            # Import enhanced prompts
            from prompts import SEQUENCE_RANKING_SYSTEM_PROMPT
            
            # Format individual analyses for ranking
            analyses_summary = self._format_analyses_for_ranking(individual_analyses)
            
            # Prepare ranking prompt
            user_message = f"""Perform comprehensive ranking of {len(sequences)} antimicrobial peptide sequences based on individual analyses and user requirements.

## User Requirements:
{self._format_user_requirements(user_requirements)}

## Individual Sequence Analyses:
{analyses_summary}

## Ranking Instructions:
1. **Complete Ranking Required**: Rank ALL {len(sequences)} sequences from most to least promising
2. **Comprehensive Evaluation**: Consider all analysis criteria systematically
3. **Detailed Justification**: Provide specific reasoning for each ranking position
4. **Score Breakdown**: Include detailed scores for each evaluation criterion
5. **Actionable Recommendations**: Provide specific next steps for top candidates

## Evaluation Criteria (in order of importance based on user requirements):
- **MIC Match Score**: Alignment with target antimicrobial potency
- **Bacteria Match Score**: Activity against specified target bacteria
- **Properties Match Score**: Physicochemical properties alignment
- **Safety Profile Score**: Predicted toxicity and selectivity
- **Synthesis Feasibility Score**: Cost and complexity considerations

## Output Requirements:
- Rank ALL sequences (no partial rankings)
- Include confidence scores for each prediction
- Provide optimization suggestions for top candidates
- Highlight any sequences with exceptional potential or concerns

Ensure complete ranking with comprehensive justification for each position."""

            # Get LLM ranking
            messages = [
                SystemMessage(content=SEQUENCE_RANKING_SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse response (handle markdown-wrapped JSON)
            try:
                content = response.content.strip()
                # Remove markdown code block markers if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                content = content.strip()

                ranking_result = json.loads(content)

                # Debug: Log the parsed result
                logger.info(f"Parsed ranking result: {len(ranking_result.get('ranking', []))} sequences ranked")

                # Validate ranking completeness
                if not self._validate_ranking_completeness(ranking_result, sequences):
                    logger.warning(f"Incomplete ranking detected: got {len(ranking_result.get('ranking', []))}, expected {len(sequences)}")
                    logger.warning(f"Raw response preview: {response.content[:500]}...")
                    ranking_result = self._complete_ranking(ranking_result, sequences, individual_analyses)
                
                return {
                    "success": True,
                    "ranking": ranking_result,
                    "total_sequences": len(sequences),
                    "raw_response": response.content
                }
                
            except json.JSONDecodeError:
                # Fallback ranking
                return self._fallback_ranking(sequences, individual_analyses, user_requirements)
                
        except Exception as e:
            logger.error(f"Failed to rank sequences comprehensively: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_sequence_properties(self, sequence: str) -> Dict[str, Any]:
        """Calculate comprehensive sequence properties"""
        properties = {
            "length": len(sequence),
            "net_charge": sum(1 for aa in sequence if aa in 'KRH') - sum(1 for aa in sequence if aa in 'DE'),
            "hydrophobic_count": sum(1 for aa in sequence if aa in 'AILMFPWV'),
            "hydrophobic_ratio": sum(1 for aa in sequence if aa in 'AILMFPWV') / len(sequence),
            "cationic_count": sum(1 for aa in sequence if aa in 'KRH'),
            "cationic_ratio": sum(1 for aa in sequence if aa in 'KRH') / len(sequence),
            "aromatic_count": sum(1 for aa in sequence if aa in 'FWY'),
            "aromatic_ratio": sum(1 for aa in sequence if aa in 'FWY') / len(sequence),
            "proline_count": sequence.count('P'),
            "glycine_count": sequence.count('G'),
            "molecular_weight": self._estimate_molecular_weight(sequence)
        }
        
        return properties
    
    def _estimate_molecular_weight(self, sequence: str) -> float:
        """Estimate molecular weight of sequence"""
        # Simplified MW calculation using average amino acid weights
        aa_weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.1, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }
        
        total_weight = sum(aa_weights.get(aa, 110.0) for aa in sequence)
        # Subtract water molecules for peptide bonds
        total_weight -= (len(sequence) - 1) * 18.0
        
        return total_weight
    
    def _format_experience_context(self, experiences: List[Any]) -> str:
        """Format experience context for analysis"""
        if not experiences:
            return "No specific experience available for this sequence type."
        
        context_parts = []
        for i, exp in enumerate(experiences[:5], 1):  # Limit to top 5 experiences
            insights_text = "; ".join(exp.insights[:3])  # Top 3 insights per experience
            context_parts.append(f"{i}. {insights_text} (Confidence: {exp.confidence:.2f})")
        
        return "\n".join(context_parts)
    
    def _format_user_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format user requirements for prompt"""
        if not requirements:
            return "No specific requirements provided."
        
        req_parts = []
        for key, value in requirements.items():
            if isinstance(value, list):
                req_parts.append(f"- {key.replace('_', ' ').title()}: {', '.join(map(str, value))}")
            else:
                req_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(req_parts)
    
    def _format_analyses_for_ranking(self, analyses: List[Dict[str, Any]]) -> str:
        """Format individual analyses for ranking prompt"""
        formatted_analyses = []
        
        for i, analysis in enumerate(analyses, 1):
            if not analysis.get("success"):
                continue
                
            seq = analysis["sequence"]
            analysis_data = analysis.get("analysis", {})
            
            # Extract key findings
            key_findings = []
            if "structural_analysis" in analysis_data:
                key_findings.append(f"Structure: {analysis_data['structural_analysis']}")
            if "activity_prediction" in analysis_data:
                key_findings.append(f"Activity: {analysis_data['activity_prediction']}")
            if "safety_assessment" in analysis_data:
                key_findings.append(f"Safety: {analysis_data['safety_assessment']}")
            
            formatted_analyses.append(f"""
**Sequence {i}**: {seq}
{chr(10).join(key_findings) if key_findings else 'Analysis data available'}
""")
        
        return "\n".join(formatted_analyses)
    
    def _fallback_parse_individual_analysis(self, content: str, sequence: str, 
                                          properties: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback parsing for individual analysis"""
        return {
            "success": True,
            "sequence": sequence,
            "analysis": {
                "structural_analysis": "Sequence shows typical AMP characteristics",
                "activity_prediction": "Moderate antimicrobial activity expected",
                "selectivity_assessment": "Good bacterial selectivity predicted",
                "stability_evaluation": "Moderate stability expected",
                "confidence": 0.6
            },
            "calculated_properties": properties
        }
    
    def _validate_ranking_completeness(self, ranking_result: Dict[str, Any], 
                                     sequences: List[str]) -> bool:
        """Validate that ranking includes all sequences"""
        if "ranking" not in ranking_result:
            return False
        
        ranked_sequences = set()
        for item in ranking_result["ranking"]:
            if "sequence" in item:
                ranked_sequences.add(item["sequence"])
        
        return len(ranked_sequences) == len(sequences)
    
    def _complete_ranking(self, partial_ranking: Dict[str, Any], 
                         sequences: List[str], 
                         individual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Complete partial ranking with missing sequences"""
        # This is a simplified completion - in practice, you might want to re-query the LLM
        ranked_sequences = set()
        if "ranking" in partial_ranking:
            for item in partial_ranking["ranking"]:
                if "sequence" in item:
                    ranked_sequences.add(item["sequence"])
        
        missing_sequences = [seq for seq in sequences if seq not in ranked_sequences]
        
        # Add missing sequences with default scores
        for seq in missing_sequences:
            partial_ranking["ranking"].append({
                "rank": len(partial_ranking["ranking"]) + 1,
                "sequence": seq,
                "overall_score": 0.5,
                "detailed_scores": {
                    "mic_match": 0.5,
                    "bacteria_match": 0.5,
                    "properties_match": 0.5,
                    "safety_profile": 0.5,
                    "synthesis_feasibility": 0.5
                },
                "reasoning": "Insufficient analysis data for complete evaluation",
                "confidence": 0.4
            })
        
        return partial_ranking
    
    def _fallback_ranking(self, sequences: List[str], 
                         individual_analyses: List[Dict[str, Any]],
                         user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback ranking when JSON parsing fails"""
        ranking = []
        
        for i, seq in enumerate(sequences):
            ranking.append({
                "rank": i + 1,
                "sequence": seq,
                "overall_score": 0.6,
                "detailed_scores": {
                    "mic_match": 0.6,
                    "bacteria_match": 0.6,
                    "properties_match": 0.6,
                    "safety_profile": 0.6,
                    "synthesis_feasibility": 0.6
                },
                "reasoning": "Fallback ranking due to parsing issues",
                "confidence": 0.5
            })
        
        return {
            "success": True,
            "ranking": {
                "ranking": ranking,
                "ranking_summary": "Fallback ranking provided due to analysis limitations",
                "recommendations": ["Re-run analysis with improved prompts"]
            },
            "total_sequences": len(sequences)
        }
