"""
Sequence Priority Ranking Module

Implements priority ranking functionality for new sequences based on user requirements
(MIC values, target bacteria, physicochemical properties).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging
from openai import OpenAI
import json
import re

from configuration import OPENAI_API_KEY, CHAT_MODEL
from experience_manager import ExperienceManager
from sequence_analyzer import SequenceAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceRanker:
    """Sequence Ranker"""

    def __init__(self):
        """Initialize sequence ranker"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.experience_manager = ExperienceManager()
        self.analyzer = SequenceAnalyzer()

    def rank_sequences(self, sequences: List[str], user_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank sequences based on user requirements

        Args:
            sequences: List of sequences to be ranked
            user_requirements: User requirements containing:
                - target_mic: Target MIC value range (min, max)
                - target_bacteria: Target bacteria list
                - desired_properties: Desired physicochemical properties
                - priority_weights: Weights for each factor

        Returns:
            Ranked sequence list, each element contains sequence and scoring information
        """
        logger.info(f"Starting to rank {len(sequences)} sequences")
        logger.info(f"User requirements: {user_requirements}")

        # Analyze each sequence
        sequence_analyses = []
        for i, seq in enumerate(sequences):
            logger.info(f"Analyzing sequence {i+1}/{len(sequences)}: {seq[:20]}...")

            analysis = self._analyze_single_sequence(seq, user_requirements)
            sequence_analyses.append({
                'sequence': seq,
                'analysis': analysis,
                'index': i
            })

        # Use GPT-4o for comprehensive ranking
        ranking_result = self._gpt_rank_sequences(sequence_analyses, user_requirements)

        return ranking_result
    
    def _analyze_single_sequence(self, sequence: str, user_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze single sequence"""

        # Basic sequence analysis
        basic_analysis = self.analyzer._analyze_composition(sequence)

        # Get relevant experiences
        query = f"sequence analysis {sequence[:20]} MIC {user_requirements.get('target_mic', '')} bacteria {user_requirements.get('target_bacteria', '')}"
        relevant_experiences = self.experience_manager.retrieve_relevant_experiences(query, top_k=5)

        # Predict physicochemical properties (based on experience)
        predicted_properties = self._predict_properties_from_experience(sequence, relevant_experiences)

        # Predict MIC range (based on experience)
        predicted_mic = self._predict_mic_from_experience(sequence, relevant_experiences)

        # Predict bacterial activity (based on experience)
        predicted_bacteria_activity = self._predict_bacteria_activity_from_experience(sequence, relevant_experiences)

        return {
            'basic_analysis': basic_analysis,
            'predicted_properties': predicted_properties,
            'predicted_mic': predicted_mic,
            'predicted_bacteria_activity': predicted_bacteria_activity,
            'relevant_experiences': [(exp.content, similarity) for exp, similarity in relevant_experiences[:3]]
        }

    def _predict_properties_from_experience(self, sequence: str, experiences: List[Tuple]) -> Dict[str, Any]:
        """Predict physicochemical properties based on experience"""
        if not experiences:
            return {}

        # Collect physicochemical property data from relevant experiences
        properties_data = []
        for exp, similarity in experiences:
            if hasattr(exp, 'properties_analysis') and exp.properties_analysis:
                for prop_key, prop_data in exp.properties_analysis.items():
                    if isinstance(prop_data, dict) and 'properties1' in prop_data:
                        properties_data.append((prop_data, similarity))

        # Predict based on similarity-weighted average
        predicted = {}
        if properties_data:
            # Simplified prediction: take average of most similar experience
            most_similar = max(properties_data, key=lambda x: x[1])
            predicted = most_similar[0].get('properties1', {})

        return predicted
    
    def _predict_mic_from_experience(self, sequence: str, experiences: List[Tuple]) -> Dict[str, Any]:
        """Predict MIC based on experience"""
        if not experiences:
            return {}

        mic_values = []
        for exp, similarity in experiences:
            if hasattr(exp, 'mic_analysis') and exp.mic_analysis:
                for mic_key, mic_data in exp.mic_analysis.items():
                    if isinstance(mic_data, dict) and 'mic1' in mic_data:
                        mic_info = mic_data['mic1']
                        if 'mean_mic' in mic_info:
                            mic_values.append((mic_info['mean_mic'], similarity))

        if mic_values:
            # Weighted average prediction
            weighted_sum = sum(mic * sim for mic, sim in mic_values)
            weight_sum = sum(sim for _, sim in mic_values)
            predicted_mic = weighted_sum / weight_sum if weight_sum > 0 else 0

            return {
                'predicted_mean_mic': predicted_mic,
                'confidence': weight_sum / len(mic_values) if mic_values else 0
            }

        return {}
    
    def _predict_bacteria_activity_from_experience(self, sequence: str, experiences: List[Tuple]) -> Dict[str, Any]:
        """Predict bacterial activity based on experience"""
        if not experiences:
            return {}

        bacteria_activity = {}
        for exp, similarity in experiences:
            if hasattr(exp, 'bacteria_analysis') and exp.bacteria_analysis:
                for bacteria_key, bacteria_data in exp.bacteria_analysis.items():
                    if isinstance(bacteria_data, dict) and 'bacteria1' in bacteria_data:
                        bacteria_info = bacteria_data['bacteria1']
                        if 'bacteria_types' in bacteria_info:
                            for bacteria, count in bacteria_info['bacteria_types'].items():
                                if bacteria not in bacteria_activity:
                                    bacteria_activity[bacteria] = []
                                bacteria_activity[bacteria].append((count, similarity))

        # Calculate predicted activity for each bacteria
        predicted_activity = {}
        for bacteria, activity_data in bacteria_activity.items():
            weighted_sum = sum(count * sim for count, sim in activity_data)
            weight_sum = sum(sim for _, sim in activity_data)
            predicted_activity[bacteria] = weighted_sum / weight_sum if weight_sum > 0 else 0

        return predicted_activity
    
    def _gpt_rank_sequences(self, sequence_analyses: List[Dict[str, Any]],
                           user_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use GPT-4o to rank sequences"""

        # Build ranking prompt
        prompt = self._build_ranking_prompt(sequence_analyses, user_requirements)

        # Call GPT-4o
        ranking_result = self._call_gpt_ranking(prompt)

        # Parse ranking results
        parsed_ranking = self._parse_ranking_result(ranking_result, sequence_analyses)

        return parsed_ranking

    def _build_ranking_prompt(self, sequence_analyses: List[Dict[str, Any]],
                             user_requirements: Dict[str, Any]) -> str:
        """Build ranking prompt"""

        # Prepare sequence information
        sequences_info = ""
        for i, seq_analysis in enumerate(sequence_analyses):
            seq = seq_analysis['sequence']
            analysis = seq_analysis['analysis']

            sequences_info += f"\n### Sequence {i+1}: {seq}\n"
            sequences_info += f"- Length: {len(seq)}\n"
            sequences_info += f"- Hydrophobic ratio: {analysis['basic_analysis']['hydrophobic_ratio']:.3f}\n"
            sequences_info += f"- Charged ratio: {analysis['basic_analysis']['charged_ratio']:.3f}\n"
            sequences_info += f"- Aromatic ratio: {analysis['basic_analysis']['aromatic_ratio']:.3f}\n"

            if analysis['predicted_mic']:
                sequences_info += f"- Predicted MIC: {analysis['predicted_mic'].get('predicted_mean_mic', 'N/A')}\n"

            if analysis['predicted_bacteria_activity']:
                top_bacteria = sorted(analysis['predicted_bacteria_activity'].items(),
                                    key=lambda x: x[1], reverse=True)[:3]
                sequences_info += f"- Main active bacteria: {', '.join([b for b, _ in top_bacteria])}\n"

            if analysis['relevant_experiences']:
                sequences_info += f"- Relevant experiences: {len(analysis['relevant_experiences'])}\n"
        
        prompt = f"""
You are a professional antimicrobial peptide design expert. Please rank the following sequences based on user requirements.

## User Requirements
- Target MIC range: {user_requirements.get('target_mic', 'Not specified')}
- Target bacteria: {user_requirements.get('target_bacteria', 'Not specified')}
- Desired properties: {user_requirements.get('desired_properties', 'Not specified')}
- Priority weights: {user_requirements.get('priority_weights', 'Not specified')}

## Sequences to Rank
{sequences_info}

## Ranking Requirements
Please evaluate and rank sequences based on the following factors:

1. **MIC Match**: How well predicted MIC values match user targets
2. **Bacteria Activity Match**: Predicted activity against target bacteria
3. **Properties Match**: How well physicochemical properties match user expectations
4. **Experience Support**: Quantity and quality of relevant experiences
5. **Sequence Feasibility**: Biological reasonableness and feasibility of sequences

Please provide for each sequence:
- Overall score (0-100)
- Detailed explanation of each score
- Strengths and weaknesses analysis
- Improvement suggestions

Please return ranking results in JSON format with the following structure:
{{
    "ranking": [
        {{
            "rank": 1,
            "sequence_index": 0,
            "sequence": "sequence",
            "overall_score": 85,
            "detailed_scores": {{
                "mic_match": 80,
                "bacteria_match": 90,
                "properties_match": 85,
                "experience_support": 75,
                "sequence_feasibility": 88
            }},
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "improvement_suggestions": ["suggestion1", "suggestion2"],
            "confidence": 0.85
        }}
    ],
    "summary": "Ranking summary and recommendations"
}}
"""
        
        return prompt

    def _call_gpt_ranking(self, prompt: str) -> str:
        """Call GPT-4o for ranking"""
        try:
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"GPT ranking call failed: {str(e)}")
            return ""

    def _get_system_prompt(self) -> str:
        """Get system prompt from prompts module"""
        try:
            from prompts import SEQUENCE_RANKING_SYSTEM_PROMPT
            return SEQUENCE_RANKING_SYSTEM_PROMPT
        except ImportError:
            return "You are a professional antimicrobial peptide design expert with extensive experience in sequence analysis and drug design."

    def _parse_ranking_result(self, ranking_result: str,
                             sequence_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse ranking results"""
        try:
            # Try to extract JSON part
            json_match = re.search(r'\{.*\}', ranking_result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Validate and supplement ranking results
                ranking = parsed.get('ranking', [])

                # Ensure all sequences are included
                included_indices = set(item.get('sequence_index', -1) for item in ranking)
                for i, seq_analysis in enumerate(sequence_analyses):
                    if i not in included_indices:
                        # Add missing sequences with default scores
                        ranking.append({
                            "rank": len(ranking) + 1,
                            "sequence_index": i,
                            "sequence": seq_analysis['sequence'],
                            "overall_score": 50,
                            "detailed_scores": {
                                "mic_match": 50,
                                "bacteria_match": 50,
                                "properties_match": 50,
                                "experience_support": 50,
                                "sequence_feasibility": 50
                            },
                            "strengths": ["Requires further analysis"],
                            "weaknesses": ["Insufficient data"],
                            "improvement_suggestions": ["Needs more experimental validation"],
                            "confidence": 0.3
                        })

                # Re-sort by score
                ranking.sort(key=lambda x: x.get('overall_score', 0), reverse=True)

                # Update rankings
                for i, item in enumerate(ranking):
                    item['rank'] = i + 1

                return ranking

            else:
                # If no JSON found, create default ranking
                return self._create_default_ranking(sequence_analyses)

        except Exception as e:
            logger.error(f"Failed to parse ranking results: {str(e)}")
            return self._create_default_ranking(sequence_analyses)

    def _create_default_ranking(self, sequence_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create default ranking"""
        ranking = []

        for i, seq_analysis in enumerate(sequence_analyses):
            # Calculate score based on simple rules
            basic_score = self._calculate_basic_score(seq_analysis)

            ranking.append({
                "rank": i + 1,
                "sequence_index": i,
                "sequence": seq_analysis['sequence'],
                "overall_score": basic_score,
                "detailed_scores": {
                    "mic_match": basic_score,
                    "bacteria_match": basic_score,
                    "properties_match": basic_score,
                    "experience_support": len(seq_analysis['analysis']['relevant_experiences']) * 10,
                    "sequence_feasibility": basic_score
                },
                "strengths": ["Based on basic analysis"],
                "weaknesses": ["Needs more detailed evaluation"],
                "improvement_suggestions": ["Recommend experimental validation"],
                "confidence": 0.5
            })

        # Sort by score
        ranking.sort(key=lambda x: x['overall_score'], reverse=True)

        # Update rankings
        for i, item in enumerate(ranking):
            item['rank'] = i + 1

        return ranking

    def _calculate_basic_score(self, seq_analysis: Dict[str, Any]) -> float:
        """Calculate basic score"""
        analysis = seq_analysis['analysis']
        basic = analysis['basic_analysis']

        # Calculate score based on heuristic rules
        score = 50  # Base score

        # Moderate hydrophobic ratio bonus
        hydrophobic_ratio = basic['hydrophobic_ratio']
        if 0.3 <= hydrophobic_ratio <= 0.7:
            score += 10

        # Moderate charged amino acid ratio bonus
        charged_ratio = basic['charged_ratio']
        if 0.1 <= charged_ratio <= 0.4:
            score += 10

        # Relevant experience bonus
        if analysis['relevant_experiences']:
            score += len(analysis['relevant_experiences']) * 5

        # MIC prediction bonus
        if analysis['predicted_mic']:
            score += 10

        # Bacterial activity prediction bonus
        if analysis['predicted_bacteria_activity']:
            score += 10

        return min(score, 100)  # Maximum 100 points

    def generate_ranking_report(self, ranking_result: List[Dict[str, Any]],
                               user_requirements: Dict[str, Any]) -> str:
        """Generate ranking report"""

        report = f"""
# Antimicrobial Peptide Sequence Priority Ranking Report

## User Requirements
- Target MIC range: {user_requirements.get('target_mic', 'Not specified')}
- Target bacteria: {user_requirements.get('target_bacteria', 'Not specified')}
- Desired properties: {user_requirements.get('desired_properties', 'Not specified')}

## Ranking Results

"""

        for item in ranking_result[:10]:  # Show top 10
            report += f"""
### Rank {item['rank']}: {item['sequence'][:30]}...

**Overall Score**: {item['overall_score']}/100
**Confidence**: {item['confidence']:.2f}

**Detailed Scores**:
- MIC Match: {item['detailed_scores']['mic_match']}/100
- Bacteria Activity Match: {item['detailed_scores']['bacteria_match']}/100
- Properties Match: {item['detailed_scores']['properties_match']}/100
- Experience Support: {item['detailed_scores']['experience_support']}/100
- Sequence Feasibility: {item['detailed_scores']['sequence_feasibility']}/100

**Strengths**: {', '.join(item['strengths'])}
**Weaknesses**: {', '.join(item['weaknesses'])}
**Improvement Suggestions**: {', '.join(item['improvement_suggestions'])}

---
"""

        return report
