"""
Enhanced AMP Designer Agent Training Module - Simplified Version

This module implements improved training functionality that processes cluster files
containing multiple similar sequences to extract comparative insights.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass
import glob

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configuration import OPENAI_API_KEY, CHAT_MODEL, CLUSTERS_DIR
from graph.experience_manager import ExperienceManager

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration parameters"""
    max_clusters_per_session: int = 20
    min_cluster_size: int = 3
    max_cluster_size: int = 50
    min_sequence_length: int = 8
    max_sequence_length: int = 60
    batch_size: int = 5
    confidence_threshold: float = 0.7

class EnhancedAMPTrainingModule:
    """Enhanced AMP Designer Agent Training Module"""

    def __init__(self, config: Optional[EnhancedTrainingConfig] = None):
        """Initialize the enhanced training module"""
        self.config = config or EnhancedTrainingConfig()

        # Initialize components
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )

        self.experience_manager = ExperienceManager()

        # Training statistics
        self.training_stats = {
            "total_clusters_analyzed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "experiences_created": 0,
            "training_sessions": 0
        }

        # Track clusters that failed to produce usable summarized insights
        self.unsummarized_clusters: List[Dict[str, Any]] = []

        logger.info("Enhanced AMP Training Module initialized successfully")
    
    def load_cluster_files(self, cluster_dir: str = None) -> List[str]:
        """Load cluster file paths from directory"""
        try:
            cluster_dir = cluster_dir or CLUSTERS_DIR
            
            # Look for cluster files
            cluster_files = glob.glob(f"{cluster_dir}/cluster_*.csv")
            
            logger.info(f"Found {len(cluster_files)} cluster files")
            return cluster_files
            
        except Exception as e:
            logger.error(f"Failed to load cluster files: {e}")
            return []
    
    def load_cluster_data(self, cluster_file: str) -> pd.DataFrame:
        """Load and validate cluster data from file"""
        try:
            df = pd.read_csv(cluster_file)
            
            # Ensure required columns exist
            if 'sequence' not in df.columns:
                logger.warning(f"No 'sequence' column in {cluster_file}")
                return pd.DataFrame()
            
            # Filter by sequence length
            df_filtered = df[
                (df['sequence'].str.len() >= self.config.min_sequence_length) &
                (df['sequence'].str.len() <= self.config.max_sequence_length)
            ].dropna(subset=['sequence'])
            
            # Filter by cluster size
            if len(df_filtered) < self.config.min_cluster_size:
                logger.info(f"Cluster {cluster_file} too small ({len(df_filtered)} sequences)")
                return pd.DataFrame()
            
            if len(df_filtered) > self.config.max_cluster_size:
                logger.info(f"Sampling {self.config.max_cluster_size} from large cluster {cluster_file}")
                df_filtered = df_filtered.sample(n=self.config.max_cluster_size, random_state=42)
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Failed to load cluster data from {cluster_file}: {e}")
            return pd.DataFrame()
    
    def analyze_sequence_cluster(self, cluster_df: pd.DataFrame, cluster_id: str) -> Dict[str, Any]:
        """Analyze multiple sequences within a cluster to extract insights"""
        try:
            from prompts import TRAINING_ANALYSIS_SYSTEM_PROMPT
            
            sequences = cluster_df['sequence'].unique().tolist()
            
            if len(sequences) < 2:
                return {"success": False, "error": "Insufficient unique sequences"}
            
            # Prepare cluster analysis: compute per-sequence stats and ranges
            per_seq = []
            lengths, charges, hydros = [], [], []
            for seq in sequences[:50]:  # cap for prompt size
                charge = self._calculate_net_charge(seq)
                hydro = self._calculate_hydrophobic_ratio(seq)
                lengths.append(len(seq)); charges.append(charge); hydros.append(hydro)
                per_seq.append(f"- {seq} (Len {len(seq)}, Charge {charge:+d}, Hydrophobic {hydro:.2f})")
            length_range = (min(lengths), max(lengths)) if lengths else (0, 0)
            charge_range = (min(charges), max(charges)) if charges else (0, 0)
            hydro_range = (float(min(hydros)), float(max(hydros))) if hydros else (0.0, 0.0)

            user_message = f"""Analyze this cluster of {len(sequences)} similar antimicrobial peptide sequences to extract detailed, transferable insights.

## Cluster Sequences (analyze each for patterns):
{chr(10).join(per_seq)}

## Property Ranges:
- Length range: {length_range}
- Net charge range: {charge_range}
- Hydrophobic ratio range: {hydro_range}

## Detailed Analysis Instructions:
1. **Position-by-position comparison**: Identify which positions vary across sequences and what substitutions occur (e.g., "Position 3: K→R substitution", "C-terminal region: hydrophobic clustering varies").

2. **Motif identification**: Look for recurring patterns like KK, RW, LW, WP, GG and note their positional context and likely functional role.

3. **Charge distribution patterns**: Analyze how cationic residues (K, R, H) are distributed - clustered vs dispersed, N/C-terminal vs central.

4. **Hydrophobic patterning**: Identify hydrophobic blocks, alternating patterns, terminal hydrophobic segments and their likely membrane interaction effects.

5. **Aromatic residue placement**: Note Trp, Phe, Tyr positions and their role in membrane anchoring or activity.

6. **Length-activity relationships**: For different lengths in this cluster, infer activity/selectivity trade-offs.

7. **Structure-function correlations**: Connect observed substitutions to likely effects on amphipathicity, membrane disruption, stability, or selectivity.

CRITICAL: Provide 8-12 detailed insights for both position_specific_insights and sequence_related_insights. Avoid generic statements. Be specific about positions, substitutions, and their effects.

Return ONLY the required JSON object with detailed insights."""

            # Get LLM analysis
            messages = [
                SystemMessage(content=TRAINING_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response (handle markdown-wrapped JSON)
            try:
                content = response.content.strip()
                # Remove markdown code block markers if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove ```
                content = content.strip()

                analysis_result = json.loads(content)

                # Basic validation and fill-ins
                if "cluster_sequences" not in analysis_result:
                    analysis_result["cluster_sequences"] = sequences
                if "sequence_summary" not in analysis_result:
                    analysis_result["sequence_summary"] = {
                        "count": len(sequences),
                        "length_range": [length_range[0], length_range[1]],
                        "net_charge_range": [charge_range[0], charge_range[1]],
                        "hydrophobic_ratio_range": [round(hydro_range[0], 3), round(hydro_range[1], 3)]
                    }
                for key in ["position_specific_insights", "sequence_related_insights", "design_principles", "contradictions_or_uncertainties"]:
                    if key not in analysis_result:
                        analysis_result[key] = []
                if "confidence" not in analysis_result:
                    analysis_result["confidence"] = 0.6

                # Track empty/unsummarized
                if (not analysis_result["position_specific_insights"] and
                    not analysis_result["sequence_related_insights"] and
                    not analysis_result["design_principles"]):
                    self.unsummarized_clusters.append({
                        "cluster_id": cluster_id,
                        "reason": "Empty insights",
                        "sequences": sequences[:10],
                        "note": "+more hidden" if len(sequences) > 10 else ""
                    })

                return {
                    "success": True,
                    "parsed_ok": True,
                    "analysis": analysis_result,
                    "cluster_info": {
                        "cluster_id": cluster_id,
                        "sequence_count": len(sequences),
                        "sequences": sequences,
                        "length_range": length_range,
                        "charge_range": charge_range,
                        "hydro_range": hydro_range
                    },
                    "raw_response": response.content
                }
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.warning(f"Raw response content: {response.content[:500]}...")
                # Mark as unsummarized due to JSON parse failure
                self.unsummarized_clusters.append({
                    "cluster_id": cluster_id,
                    "reason": "JSON parse failure",
                    "sequences": sequences[:10],
                    "note": "+more hidden" if len(sequences) > 10 else "",
                    "raw_response_preview": response.content[:200]
                })
                return {
                    "success": True,
                    "parsed_ok": False,
                    "analysis": {
                        "position_specific_insights": [],
                        "sequence_related_insights": [],
                        "design_principles": [],
                        "contradictions_or_uncertainties": [],
                        "confidence": 0.5
                    },
                    "cluster_info": {
                        "cluster_id": cluster_id,
                        "sequence_count": len(sequences),
                        "sequences": sequences,
                        "length_range": length_range,
                        "charge_range": charge_range,
                        "hydro_range": hydro_range
                    },
                    "raw_response": response.content
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze sequence cluster: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_net_charge(self, sequence: str) -> int:
        """Calculate net charge of a sequence"""
        positive = sum(1 for aa in sequence if aa in 'KRH')
        negative = sum(1 for aa in sequence if aa in 'DE')
        return positive - negative
    
    def _calculate_hydrophobic_ratio(self, sequence: str) -> float:
        """Calculate hydrophobic amino acid ratio"""
        hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV')
        return hydrophobic / len(sequence) if sequence else 0.0
    
    def store_cluster_experience(self, cluster_analysis: Dict[str, Any]) -> Optional[str]:
        """Store cluster analysis experience"""
        try:
            if not cluster_analysis.get("success"):
                return None

            analysis = cluster_analysis["analysis"]
            cluster_info = cluster_analysis.get("cluster_info", {})

            # Extract insights
            insights = []
            if "position_specific_insights" in analysis:
                insights.extend(analysis["position_specific_insights"])
            if "design_principles" in analysis:
                insights.extend(analysis["design_principles"])

            # Create experience content (remove file/relative info; dedup by insight text)
            sequence_count = cluster_info.get("sequence_count", 0)
            sequences = cluster_info.get("sequences", [])
            length_range = cluster_info.get("length_range", (0, 0))
            charge_range = cluster_info.get("charge_range", (0, 0))
            hydro_range = cluster_info.get("hydro_range", (0.0, 0.0))

            # Compose content primarily from insights to aid deduplication
            primary_text = "; ".join(list(dict.fromkeys(insights)))[:1000] or "Sequence-related AMP design insights"
            content = primary_text
            # Compact context without filenames/IDs; include brief sequence summary
            seq_preview = "; ".join(sequences[:5]) + ("; +" + str(len(sequences)-5) + " more" if len(sequences) > 5 else "")
            context = (
                f"Derived from comparative analysis of {sequence_count} similar AMPs "
                f"(len {length_range[0]}-{length_range[1]}, charge {charge_range[0]}-{charge_range[1]}, "
                f"hydrophobic {hydro_range[0]:.2f}-{hydro_range[1]:.2f}). Sequences: {seq_preview}"
            )

            # Store experience with required sequence_pair parameter
            sequence_pair = (sequences[0], sequences[1]) if len(sequences) >= 2 else ("", "")

            exp_id = self.experience_manager.add_experience(
                content=content,
                context=context,
                sequence_pair=sequence_pair,
                insights=insights,
                confidence=analysis.get("confidence", 0.7)
            )

            if exp_id:
                self.training_stats["experiences_created"] += 1
                logger.info(f"Stored cluster experience: {exp_id}")

            return exp_id

        except Exception as e:
            logger.error(f"Failed to store cluster experience: {e}")
            return None
    
    def run_enhanced_training_session(self, max_clusters: Optional[int] = None,
                                    cluster_dir: str = None) -> Dict[str, Any]:
        """Run enhanced training session processing cluster files"""
        try:
            session_start = datetime.now()
            max_clusters = max_clusters or self.config.max_clusters_per_session
            
            logger.info(f"Starting enhanced training session with max {max_clusters} clusters")
            
            # Load cluster files
            cluster_files = self.load_cluster_files(cluster_dir)
            
            if not cluster_files:
                return {"success": False, "error": "No cluster files found"}
            
            # Limit clusters if specified
            if max_clusters and len(cluster_files) > max_clusters:
                cluster_files = cluster_files[:max_clusters]
            
            # Process clusters
            successful_analyses = 0
            failed_analyses = 0
            experiences_created = 0
            
            for cluster_file in cluster_files:
                try:
                    # Load cluster data
                    cluster_df = self.load_cluster_data(cluster_file)
                    
                    if cluster_df.empty:
                        continue
                    
                    # Extract cluster ID from filename
                    cluster_id = Path(cluster_file).stem
                    
                    # Analyze cluster
                    analysis_result = self.analyze_sequence_cluster(cluster_df, cluster_id)
                    
                    self.training_stats["total_clusters_analyzed"] += 1
                    
                    if analysis_result.get("success"):
                        successful_analyses += 1
                        self.training_stats["successful_analyses"] += 1
                        
                        # Store experience
                        exp_id = self.store_cluster_experience(analysis_result)
                        if exp_id:
                            experiences_created += 1
                    else:
                        failed_analyses += 1
                        self.training_stats["failed_analyses"] += 1
                
                except Exception as e:
                    failed_analyses += 1
                    self.training_stats["failed_analyses"] += 1
                    logger.error(f"Error processing cluster {cluster_file}: {e}")
            
            # Save experiences
            self.experience_manager.save_experiences()
            self.training_stats["training_sessions"] += 1
            
            session_duration = (datetime.now() - session_start).total_seconds()
            
            return {
                "success": True,
                "session_stats": {
                    "total_clusters_processed": len(cluster_files),
                    "successful_analyses": successful_analyses,
                    "failed_analyses": failed_analyses,
                    "experiences_created": experiences_created,
                    "session_duration_seconds": session_duration
                },
                "overall_stats": self.training_stats.copy(),
                "unsummarized": self.unsummarized_clusters.copy()
            }
            
        except Exception as e:
            logger.error(f"Enhanced training session failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_enhanced_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced training statistics"""
        try:
            exp_stats = self.experience_manager.get_statistics()
            
            return {
                "training_stats": self.training_stats.copy(),
                "experience_stats": exp_stats,
                "config": {
                    "max_clusters_per_session": self.config.max_clusters_per_session,
                    "min_cluster_size": self.config.min_cluster_size,
                    "max_cluster_size": self.config.max_cluster_size,
                    "batch_size": self.config.batch_size,
                    "confidence_threshold": self.config.confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get enhanced training statistics: {e}")
            return {"error": str(e)}
