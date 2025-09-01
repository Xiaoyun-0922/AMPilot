"""Main processing functions for clustering and dereplication."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from collections import Counter

from ..ingest_normalize.schemas import Sequence, SequenceBatch
from ..homology_search.schemas import HomologSet, SearchResult
from .schemas import (
    ClusterSet, DedupBatch, ClusteringStats, Cluster, ClusterMember,
    ClusteringMethod, ClusteringMode
)
from .cdhit_engine import CDHitEngine


def cluster_sequences(
    input_sequences: Union[List[Sequence], SequenceBatch, HomologSet, str, Path],
    clustering_method: ClusteringMethod = ClusteringMethod.CD_HIT,
    identity_threshold: float = 0.9,
    coverage_threshold: float = 0.8,
    clustering_mode: ClusteringMode = ClusteringMode.HYBRID,
    num_threads: int = 1,
    cdhit_bin_path: Optional[str] = None,
    **kwargs
) -> tuple[ClusterSet, DedupBatch]:
    """
    Cluster sequences and create deduplicated representative set.
    
    Args:
        input_sequences: Input sequences (various formats supported)
        clustering_method: Clustering algorithm to use
        identity_threshold: Sequence identity threshold (0.0-1.0)
        coverage_threshold: Coverage threshold (0.0-1.0)
        clustering_mode: Clustering mode (identity, coverage, hybrid)
        num_threads: Number of threads to use
        cdhit_bin_path: Path to CD-HIT binaries
        **kwargs: Additional clustering parameters
        
    Returns:
        Tuple of (ClusterSet, DedupBatch)
    """
    
    # Parse and validate input sequences
    sequences = _parse_clustering_input(input_sequences)
    
    if not sequences:
        raise ValueError("No valid sequences provided for clustering")
    
    # Initialize clustering engine
    if clustering_method == ClusteringMethod.CD_HIT:
        engine = CDHitEngine(cdhit_bin_path)
        clusters = engine.cluster_sequences(
            sequences=sequences,
            identity_threshold=identity_threshold,
            coverage_threshold=coverage_threshold,
            num_threads=num_threads,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Clustering method {clustering_method} not yet implemented")
    
    # Create ClusterSet
    cluster_set = ClusterSet(
        clusters=clusters,
        total_clusters=len(clusters),
        total_sequences=len(sequences),
        clustering_method=clustering_method,
        clustering_mode=clustering_mode,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        metadata={
            "original_sequence_count": len(sequences),
            "clustering_stats": _calculate_clustering_stats(clusters, len(sequences))
        }
    )
    
    # Create DedupBatch with representatives
    representatives = [cluster.representative for cluster in clusters]
    cluster_mapping = {}
    
    for cluster in clusters:
        # Map representative to its own cluster
        cluster_mapping[cluster.representative.sequence_id] = cluster.cluster_id
        # Map all members to this cluster
        for member in cluster.members:
            cluster_mapping[member.sequence_id] = cluster.cluster_id
    
    dedup_batch = DedupBatch(
        representative_sequences=representatives,
        cluster_mapping=cluster_mapping,
        reduction_ratio=(len(sequences) - len(representatives)) / len(sequences) if sequences else 0,
        original_count=len(sequences),
        deduplicated_count=len(representatives),
        metadata={
            "clustering_method": clustering_method.value,
            "identity_threshold": identity_threshold,
            "coverage_threshold": coverage_threshold
        }
    )
    
    return cluster_set, dedup_batch


def _parse_clustering_input(input_data: Union[List[Sequence], SequenceBatch, HomologSet, str, Path]) -> List[Sequence]:
    """Parse various input formats to List[Sequence]."""
    
    if isinstance(input_data, list) and all(isinstance(seq, Sequence) for seq in input_data):
        return input_data
    
    elif isinstance(input_data, SequenceBatch):
        # Extract valid sequences from batch
        return [seq for seq in input_data.sequences 
                if any(flag.value == "valid" for flag in seq.quality_flags)]
    
    elif isinstance(input_data, HomologSet):
        # Extract all unique sequences from homology results
        unique_sequences = {}
        
        # Add query sequences
        for search_result in input_data.search_results:
            # We need to reconstruct query sequences from search results
            # This is a limitation - ideally we'd store query sequences in HomologSet
            pass
        
        # Add hit sequences (if available)
        for search_result in input_data.search_results:
            for hit in search_result.hits:
                if hit.target_sequence:  # If sequence is available
                    seq_id = hit.target_id
                    if seq_id not in unique_sequences:
                        unique_sequences[seq_id] = Sequence(
                            id=seq_id,
                            sequence=hit.target_sequence,
                            description=hit.target_description
                        )
        
        return list(unique_sequences.values())
    
    elif isinstance(input_data, (str, Path)):
        # Load from JSON file
        with open(input_data, 'r') as f:
            data = json.load(f)
        
        if "batch" in data:
            # SequenceBatch format
            sequences = []
            for seq_data in data["batch"]["sequences"]:
                seq = Sequence(
                    id=seq_data["id"],
                    sequence=seq_data["sequence"],
                    description=seq_data.get("description"),
                    length=seq_data.get("length")
                )
                sequences.append(seq)
            return sequences
        
        elif "homolog_set" in data:
            # HomologSet format - extract hit sequences
            sequences = {}
            for search_result in data["homolog_set"]["search_results"]:
                for hit in search_result["hits"]:
                    if hit.get("target_sequence"):
                        seq_id = hit["target_id"]
                        if seq_id not in sequences:
                            sequences[seq_id] = Sequence(
                                id=seq_id,
                                sequence=hit["target_sequence"],
                                description=hit.get("target_description")
                            )
            return list(sequences.values())
        
        else:
            raise ValueError("Invalid input file format")
    
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")


def _calculate_clustering_stats(clusters: List[Cluster], total_input: int) -> ClusteringStats:
    """Calculate clustering statistics."""
    
    if not clusters:
        return ClusteringStats(
            total_input_sequences=total_input,
            total_clusters=0,
            largest_cluster_size=0,
            smallest_cluster_size=0,
            average_cluster_size=0,
            singleton_clusters=0,
            reduction_factor=0,
            cluster_size_distribution={}
        )
    
    cluster_sizes = [cluster.size for cluster in clusters]
    
    # Size distribution
    size_distribution = {
        "1": sum(1 for size in cluster_sizes if size == 1),
        "2-5": sum(1 for size in cluster_sizes if 2 <= size <= 5),
        "6-10": sum(1 for size in cluster_sizes if 6 <= size <= 10),
        "11-50": sum(1 for size in cluster_sizes if 11 <= size <= 50),
        "51+": sum(1 for size in cluster_sizes if size > 50)
    }
    
    stats = ClusteringStats(
        total_input_sequences=total_input,
        total_clusters=len(clusters),
        largest_cluster_size=max(cluster_sizes),
        smallest_cluster_size=min(cluster_sizes),
        average_cluster_size=sum(cluster_sizes) / len(cluster_sizes),
        singleton_clusters=sum(1 for size in cluster_sizes if size == 1),
        reduction_factor=len(clusters) / total_input if total_input > 0 else 0,
        cluster_size_distribution=size_distribution
    )
    
    return stats


def analyze_cluster_composition(cluster_set: ClusterSet) -> Dict[str, Any]:
    """Analyze the composition and characteristics of clusters."""
    
    analysis = {
        "cluster_count": cluster_set.total_clusters,
        "total_sequences": cluster_set.total_sequences,
        "reduction_ratio": 1 - (cluster_set.total_clusters / cluster_set.total_sequences) if cluster_set.total_sequences > 0 else 0,
        "size_statistics": {},
        "length_statistics": {},
        "identity_distribution": {}
    }
    
    if not cluster_set.clusters:
        return analysis
    
    # Cluster size analysis
    sizes = [cluster.size for cluster in cluster_set.clusters]
    analysis["size_statistics"] = {
        "min": min(sizes),
        "max": max(sizes),
        "mean": sum(sizes) / len(sizes),
        "median": sorted(sizes)[len(sizes) // 2]
    }
    
    # Length analysis
    lengths = [cluster.avg_length for cluster in cluster_set.clusters]
    analysis["length_statistics"] = {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths)
    }
    
    # Representative sequences for downstream analysis
    analysis["representative_count"] = len(cluster_set.clusters)
    analysis["largest_clusters"] = sorted(
        [(cluster.cluster_id, cluster.size) for cluster in cluster_set.clusters],
        key=lambda x: x[1], reverse=True
    )[:10]
    
    return analysis


def extract_representatives(
    cluster_set: ClusterSet,
    min_cluster_size: int = 1,
    max_representatives: Optional[int] = None
) -> List[Sequence]:
    """
    Extract representative sequences from clusters.
    
    Args:
        cluster_set: Input cluster set
        min_cluster_size: Minimum cluster size to include
        max_representatives: Maximum number of representatives to return
        
    Returns:
        List of representative sequences
    """
    
    # Filter clusters by size
    filtered_clusters = [
        cluster for cluster in cluster_set.clusters
        if cluster.size >= min_cluster_size
    ]
    
    # Sort by cluster size (largest first)
    filtered_clusters.sort(key=lambda c: c.size, reverse=True)
    
    # Limit number if requested
    if max_representatives:
        filtered_clusters = filtered_clusters[:max_representatives]
    
    # Extract representative sequences
    representatives = []
    for cluster in filtered_clusters:
        rep = cluster.representative
        seq = Sequence(
            id=rep.sequence_id,
            sequence=rep.sequence,
            description=f"Cluster_{cluster.cluster_id}_rep (size={cluster.size})",
            length=rep.length
        )
        representatives.append(seq)
    
    return representatives
