"""
Sequence Analysis Agent Core Logic

Implements GPT-4o based sequence analysis logic, including sequence comparison analysis,
physicochemical property correlation analysis and experience accumulation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging
from openai import OpenAI
import json
import re

from configuration import OPENAI_API_KEY, CHAT_MODEL
from .experience_manager import ExperienceManager, Experience

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceAnalyzer:
    """Sequence analyzer"""

    def __init__(self):
        """Initialize sequence analyzer"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.experience_manager = ExperienceManager()

    def analyze_sequence_pair(self, seq1: str, seq2: str, data1: pd.DataFrame,
                             data2: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sequence pair and learn experience

        Args:
            seq1, seq2: Two sequences
            data1, data2: Corresponding data

        Returns:
            Analysis results
        """
        logger.info(f"Analyzing sequence pair: {seq1[:20]}... vs {seq2[:20]}...")

        # Prepare analysis data
        analysis_data = self._prepare_analysis_data(seq1, seq2, data1, data2)

        # Build analysis prompt
        prompt = self._build_analysis_prompt(analysis_data)

        # Call GPT-4o for analysis
        analysis_result = self._call_gpt_analysis(prompt)

        # Parse analysis results
        parsed_result = self._parse_analysis_result(analysis_result)

        # Store experience
        self._store_experience(seq1, seq2, analysis_data, parsed_result)

        return parsed_result

    def _prepare_analysis_data(self, seq1: str, seq2: str, data1: pd.DataFrame,
                              data2: pd.DataFrame) -> Dict[str, Any]:
        """Prepare analysis data"""

        # Calculate basic sequence information
        seq1_info = {
            'sequence': seq1,
            'length': len(seq1),
            'composition': self._analyze_composition(seq1)
        }

        seq2_info = {
            'sequence': seq2,
            'length': len(seq2),
            'composition': self._analyze_composition(seq2)
        }

        # Analyze physicochemical properties
        properties1 = self._analyze_properties(data1)
        properties2 = self._analyze_properties(data2)

        # Analyze MIC data
        mic1 = self._analyze_mic_data(data1)
        mic2 = self._analyze_mic_data(data2)

        # Analyze bacterial data
        bacteria1 = self._analyze_bacteria_data(data1)
        bacteria2 = self._analyze_bacteria_data(data2)
        
        return {
            'sequence1': seq1_info,
            'sequence2': seq2_info,
            'properties1': properties1,
            'properties2': properties2,
            'mic1': mic1,
            'mic2': mic2,
            'bacteria1': bacteria1,
            'bacteria2': bacteria2,
            'sequence_alignment': self._align_sequences(seq1, seq2)
        }
    
    def _analyze_composition(self, sequence: str) -> Dict[str, Any]:
        """Analyze sequence composition"""
        amino_acids = list(sequence)
        composition = {}

        for aa in set(amino_acids):
            composition[aa] = amino_acids.count(aa) / len(amino_acids)

        # Analyze special properties
        hydrophobic = sum(1 for aa in amino_acids if aa in 'AILMFPWV') / len(amino_acids)
        hydrophilic = sum(1 for aa in amino_acids if aa in 'NQST') / len(amino_acids)
        charged = sum(1 for aa in amino_acids if aa in 'DEKR') / len(amino_acids)
        aromatic = sum(1 for aa in amino_acids if aa in 'FWY') / len(amino_acids)

        return {
            'composition': composition,
            'hydrophobic_ratio': hydrophobic,
            'hydrophilic_ratio': hydrophilic,
            'charged_ratio': charged,
            'aromatic_ratio': aromatic
        }

    def _analyze_properties(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze physicochemical properties"""
        properties_cols = ['Z1_Lipophilicity', 'Z2_Steric_properties', 
                          'Z3_Electronic_properties', 'Z4_Electronegativity', 
                          'Z5_Thermochemical']
        
        properties = {}
        for col in properties_cols:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 0:
                    properties[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
        
        return properties
    
    def _analyze_mic_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze MIC data"""
        if 'value' in data.columns:
            mic_values = data['value'].dropna()
            if len(mic_values) > 0:
                return {
                    'mean_mic': float(mic_values.mean()),
                    'std_mic': float(mic_values.std()),
                    'min_mic': float(mic_values.min()),
                    'max_mic': float(mic_values.max()),
                    'count': len(mic_values)
                }
        return {}
    
    def _analyze_bacteria_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze bacterial data"""
        if 'bacterium' in data.columns:
            bacteria = data['bacterium'].dropna().value_counts()
            return {
                'bacteria_types': bacteria.to_dict(),
                'total_bacteria_types': len(bacteria),
                'most_common': bacteria.index[0] if len(bacteria) > 0 else None
            }
        return {}

    def _align_sequences(self, seq1: str, seq2: str) -> Dict[str, Any]:
        """Simple sequence alignment analysis"""
        # Find identical and different positions
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])

        # Find common subsequences
        common_subseqs = []
        for i in range(len(seq1)):
            for j in range(i+3, len(seq1)+1):  # At least 3 amino acid subsequences
                subseq = seq1[i:j]
                if subseq in seq2:
                    common_subseqs.append(subseq)

        return {
            'length_diff': abs(len(seq1) - len(seq2)),
            'matches_in_overlap': matches,
            'match_ratio': matches / min_len if min_len > 0 else 0,
            'common_subsequences': list(set(common_subseqs))[:10]  # Maximum 10
        }

    def _build_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Build analysis prompt"""

        # Get relevant experiences
        query = f"sequence analysis {analysis_data['sequence1']['sequence'][:20]} {analysis_data['sequence2']['sequence'][:20]}"
        relevant_experiences = self.experience_manager.retrieve_relevant_experiences(query, top_k=3)

        experience_context = ""
        if relevant_experiences:
            experience_context = "\n\n## Relevant Experience Reference:\n"
            for i, (exp, similarity) in enumerate(relevant_experiences, 1):
                experience_context += f"\n### Experience {i} (Similarity: {similarity:.3f}):\n"
                experience_context += f"Content: {exp.content}\n"
                experience_context += f"Insights: {', '.join(exp.insights)}\n"

        prompt = f"""
You are a professional antimicrobial peptide sequence analysis expert. Please analyze the following two similar antimicrobial peptide sequences and learn their characteristics and patterns.

## Sequence Information
### Sequence 1: {analysis_data['sequence1']['sequence']}
- Length: {analysis_data['sequence1']['length']}
- Hydrophobic amino acid ratio: {analysis_data['sequence1']['composition']['hydrophobic_ratio']:.3f}
- Charged amino acid ratio: {analysis_data['sequence1']['composition']['charged_ratio']:.3f}
- Aromatic amino acid ratio: {analysis_data['sequence1']['composition']['aromatic_ratio']:.3f}

### Sequence 2: {analysis_data['sequence2']['sequence']}
- Length: {analysis_data['sequence2']['length']}
- Hydrophobic amino acid ratio: {analysis_data['sequence2']['composition']['hydrophobic_ratio']:.3f}
- Charged amino acid ratio: {analysis_data['sequence2']['composition']['charged_ratio']:.3f}
- Aromatic amino acid ratio: {analysis_data['sequence2']['composition']['aromatic_ratio']:.3f}

## Sequence Alignment Information
- Length difference: {analysis_data['sequence_alignment']['length_diff']}
- Overlap region match ratio: {analysis_data['sequence_alignment']['match_ratio']:.3f}
- Common subsequences: {', '.join(analysis_data['sequence_alignment']['common_subsequences'][:5])}

## Physicochemical Properties Comparison
### Sequence 1 Properties:
{json.dumps(analysis_data['properties1'], indent=2, ensure_ascii=False)}

### Sequence 2 Properties:
{json.dumps(analysis_data['properties2'], indent=2, ensure_ascii=False)}

## MIC Data Comparison
### Sequence 1 MIC:
{json.dumps(analysis_data['mic1'], indent=2, ensure_ascii=False)}

### Sequence 2 MIC:
{json.dumps(analysis_data['mic2'], indent=2, ensure_ascii=False)}

## Bacterial Activity Comparison
### Sequence 1 Bacterial Activity:
{json.dumps(analysis_data['bacteria1'], indent=2, ensure_ascii=False)}

### Sequence 2 Bacterial Activity:
{json.dumps(analysis_data['bacteria2'], indent=2, ensure_ascii=False)}

{experience_context}

## Analysis Tasks
Please conduct in-depth analysis following these steps:

1. **Sequence Similarity Analysis**: Analyze similarities and differences between the two sequences, focusing on:
   - What do the identical sequence fragments indicate?
   - What do the different sequence fragments indicate?
   - How do these differences affect antimicrobial activity?

2. **Physicochemical Properties Correlation Analysis**: Compare physicochemical property differences between the two sequences:
   - Which physicochemical properties are similar? What does this indicate?
   - Which physicochemical properties are different? What is the significance of these differences?
   - How do physicochemical properties relate to MIC values?
   - Which sequence differences might have caused these variations?

3. **Antimicrobial Activity Analysis**: Analyze MIC differences and bacterial spectrum differences:
   - What do the MIC value differences indicate?
   - What do the activity differences against different bacteria reflect?
   - How does sequence structure affect antimicrobial spectrum?
   - Which sequence differences might have caused these variations?

4. **Structure-Function Relationship Insights**: Summarize the relationship between sequence structure and function:
   - What are the key functional regions?
   - Which amino acid positions are most important for activity?
   - How can sequences be optimized to improve activity?
   - Which sequence differences might have caused these variations?

Please return analysis results in JSON format with the following fields:
- sequence_similarity_analysis: Sequence similarity analysis
- properties_correlation_analysis: Physicochemical properties correlation analysis
- activity_analysis: Antimicrobial activity analysis
- structure_function_insights: Structure-function relationship insights
- key_findings: List of key findings
- optimization_suggestions: List of optimization suggestions
- confidence_score: Analysis confidence score (0-1)
"""

        return prompt

    def _call_gpt_analysis(self, prompt: str) -> str:
        """Call GPT-4o for analysis"""
        try:
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"GPT analysis call failed: {str(e)}")
            return ""

    def _get_system_prompt(self) -> str:
        """Get system prompt from prompts module"""
        try:
            from prompts import SEQUENCE_ANALYSIS_SYSTEM_PROMPT
            return SEQUENCE_ANALYSIS_SYSTEM_PROMPT
        except ImportError:
            return "You are a professional antimicrobial peptide sequence analysis expert with deep knowledge in biochemistry and molecular biology."

    def _parse_analysis_result(self, analysis_result: str) -> Dict[str, Any]:
        """Parse analysis results"""
        try:
            # Try to extract JSON part
            json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                return parsed
            else:
                # If no JSON found, create default structure
                return {
                    "sequence_similarity_analysis": analysis_result[:500],
                    "properties_correlation_analysis": "",
                    "activity_analysis": "",
                    "structure_function_insights": "",
                    "key_findings": [],
                    "optimization_suggestions": [],
                    "confidence_score": 0.5
                }

        except Exception as e:
            logger.error(f"Failed to parse analysis results: {str(e)}")
            return {
                "sequence_similarity_analysis": analysis_result[:500],
                "properties_correlation_analysis": "",
                "activity_analysis": "",
                "structure_function_insights": "",
                "key_findings": [],
                "optimization_suggestions": [],
                "confidence_score": 0.3
            }

    def _store_experience(self, seq1: str, seq2: str, analysis_data: Dict[str, Any],
                         parsed_result: Dict[str, Any]):
        """Store analysis experience"""
        try:
            content = f"Analyzed sequence pair {seq1[:20]}... and {seq2[:20]}... for similarities and differences"
            context = f"Sequence lengths: {len(seq1)}, {len(seq2)}; Match ratio: {analysis_data['sequence_alignment']['match_ratio']:.3f}"

            # Extract key information
            properties_analysis = {
                'properties1': analysis_data['properties1'],
                'properties2': analysis_data['properties2']
            }

            mic_analysis = {
                'mic1': analysis_data['mic1'],
                'mic2': analysis_data['mic2']
            }

            bacteria_analysis = {
                'bacteria1': analysis_data['bacteria1'],
                'bacteria2': analysis_data['bacteria2']
            }

            insights = parsed_result.get('key_findings', [])
            confidence = parsed_result.get('confidence_score', 0.5)

            # Store experience
            exp_id = self.experience_manager.add_experience(
                content=content,
                context=context,
                sequence_pair=(seq1, seq2),
                properties_analysis=properties_analysis,
                mic_analysis=mic_analysis,
                bacteria_analysis=bacteria_analysis,
                insights=insights,
                confidence=confidence
            )

            logger.info(f"Experience stored successfully: {exp_id}")

        except Exception as e:
            logger.error(f"Failed to store experience: {str(e)}")

    def save_experiences(self):
        """Save all experiences to file"""
        self.experience_manager.save_experiences()
