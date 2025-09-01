"""FastMCP wrapper for clustering and dereplication."""

import asyncio
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict

from fastmcp import FastMCP

from .main import cluster_sequences, analyze_cluster_composition, extract_representatives
from .schemas import ClusteringMethod, ClusteringMode, ClusterSet, DedupBatch


# Create FastMCP server instance
mcp_server = FastMCP("Clustering & Dereplication")


@mcp_server.tool()
async def cluster_and_dereplicate(
    input_sequences: str,
    clustering_method: str = "cd-hit",
    identity_threshold: float = 0.9,
    coverage_threshold: float = 0.8,
    clustering_mode: str = "hybrid",
    num_threads: int = 1,
    cdhit_bin_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Cluster sequences to remove redundancy and identify representative sequences.
    
    This tool groups similar sequences together and selects representative sequences
    for each cluster, effectively reducing dataset size while preserving diversity.
    
    Args:
        input_sequences: Path to input file (SequenceBatch or HomologSet JSON)
        clustering_method: Clustering algorithm - "cd-hit", "mmseqs2", "usearch" (default: "cd-hit")
        identity_threshold: Sequence identity threshold 0.0-1.0 (default: 0.9)
        coverage_threshold: Coverage threshold 0.0-1.0 (default: 0.8)
        clustering_mode: Mode - "identity", "coverage", "hybrid" (default: "hybrid")
        num_threads: Number of CPU threads (default: 1)
        cdhit_bin_path: Path to CD-HIT binaries directory (optional)
        output_path: Path to save results as JSON (optional)
        **kwargs: Additional clustering parameters (e.g., cdhit_word_size=5)
        
    Returns:
        Dictionary containing:
        - cluster_set: Detailed clustering results with all clusters
        - dedup_batch: Representative sequences for downstream analysis
        - analysis: Statistical analysis of clustering results
        - output_file: Path to saved results (if output_path provided)
    """
    
    try:
        # Parse clustering method
        try:
            method = ClusteringMethod(clustering_method.lower().replace("-", "_"))
        except ValueError:
            return {
                "error": f"Unsupported clustering method: {clustering_method}. Supported: cd-hit, mmseqs2, usearch",
                "success": False
            }
        
        # Parse clustering mode
        try:
            mode = ClusteringMode(clustering_mode.lower())
        except ValueError:
            return {
                "error": f"Unsupported clustering mode: {clustering_mode}. Supported: identity, coverage, hybrid",
                "success": False
            }
        
        # Validate input file
        if not Path(input_sequences).exists():
            return {
                "error": f"Input file not found: {input_sequences}",
                "success": False
            }
        
        # Validate thresholds
        if not (0.0 <= identity_threshold <= 1.0):
            return {
                "error": "Identity threshold must be between 0.0 and 1.0",
                "success": False
            }
        
        if not (0.0 <= coverage_threshold <= 1.0):
            return {
                "error": "Coverage threshold must be between 0.0 and 1.0",
                "success": False
            }
        
        # Run clustering
        cluster_set, dedup_batch = await asyncio.get_event_loop().run_in_executor(
            None,
            cluster_sequences,
            input_sequences,
            method,
            identity_threshold,
            coverage_threshold,
            mode,
            num_threads,
            cdhit_bin_path,
            **kwargs
        )
        
        # Analyze clustering results
        analysis = await asyncio.get_event_loop().run_in_executor(
            None,
            analyze_cluster_composition,
            cluster_set
        )
        
        # Convert to dictionaries for JSON serialization
        cluster_dict = asdict(cluster_set)
        dedup_dict = asdict(dedup_batch)
        
        # Serialize enums
        cluster_dict = _serialize_enums(cluster_dict)
        dedup_dict = _serialize_enums(dedup_dict)
        
        # Create summary
        summary = {
            "original_sequences": cluster_set.total_sequences,
            "clusters_created": cluster_set.total_clusters,
            "representative_sequences": dedup_batch.deduplicated_count,
            "reduction_ratio": dedup_batch.reduction_ratio,
            "compression_factor": f"{cluster_set.total_sequences}:{cluster_set.total_clusters}",
            "largest_cluster_size": analysis["size_statistics"]["max"] if analysis["size_statistics"] else 0,
            "singleton_clusters": analysis.get("singleton_clusters", 0)
        }
        
        result = {
            "cluster_set": cluster_dict,
            "dedup_batch": dedup_dict,
            "analysis": analysis,
            "summary": summary,
            "success": True
        }
        
        # Save to file if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            result["output_file"] = str(output_file)
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__,
            "success": False
        }


@mcp_server.tool()
async def extract_cluster_representatives(
    cluster_input: str,
    min_cluster_size: int = 1,
    max_representatives: Optional[int] = None,
    output_format: str = "json",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract representative sequences from clustering results.
    
    Get the representative sequence from each cluster for downstream analysis.
    Useful for creating non-redundant sequence sets.
    
    Args:
        cluster_input: Path to ClusterSet JSON file from previous clustering
        min_cluster_size: Minimum cluster size to include (default: 1)
        max_representatives: Maximum number of representatives to extract (optional)
        output_format: Output format - "json", "fasta" (default: "json")
        output_path: Path to save extracted representatives (optional)
        
    Returns:
        Dictionary containing representative sequences and statistics
    """
    
    try:
        # Load cluster set
        if not Path(cluster_input).exists():
            return {
                "error": f"Input file not found: {cluster_input}",
                "success": False
            }
        
        with open(cluster_input, 'r') as f:
            data = json.load(f)
        
        if "cluster_set" not in data:
            return {
                "error": "Invalid input file - missing cluster_set",
                "success": False
            }
        
        # Convert back to ClusterSet object (simplified reconstruction)
        # In a full implementation, you'd have proper deserialization
        
        # For now, work with the dictionary representation
        cluster_data = data["cluster_set"]
        
        # Extract representatives based on criteria
        representatives = []
        cluster_count = 0
        
        for cluster in cluster_data["clusters"]:
            cluster_size = cluster["size"]
            
            # Apply filters
            if cluster_size >= min_cluster_size:
                rep = cluster["representative"]
                representatives.append({
                    "id": rep["sequence_id"],
                    "sequence": rep["sequence"],
                    "description": f"Cluster_{cluster['cluster_id']}_rep (size={cluster_size})",
                    "length": rep["length"],
                    "cluster_id": cluster["cluster_id"],
                    "cluster_size": cluster_size
                })
                cluster_count += 1
                
                # Check max limit
                if max_representatives and len(representatives) >= max_representatives:
                    break
        
        # Sort by cluster size (largest first)
        representatives.sort(key=lambda x: x["cluster_size"], reverse=True)
        
        # Create result
        result = {
            "representatives": representatives,
            "count": len(representatives),
            "statistics": {
                "total_clusters_processed": cluster_count,
                "representatives_extracted": len(representatives),
                "size_range": {
                    "min": min(r["cluster_size"] for r in representatives) if representatives else 0,
                    "max": max(r["cluster_size"] for r in representatives) if representatives else 0
                }
            },
            "success": True
        }
        
        # Save to file if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format.lower() == "fasta":
                # Write FASTA format
                with open(output_file, 'w') as f:
                    for rep in representatives:
                        f.write(f">{rep['id']} {rep['description']}\n")
                        f.write(f"{rep['sequence']}\n")
            else:
                # Write JSON format
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            result["output_file"] = str(output_file)
            result["output_format"] = output_format
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__,
            "success": False
        }


@mcp_server.tool()
async def analyze_clustering_results(cluster_input: str) -> Dict[str, Any]:
    """
    Analyze clustering results and provide detailed statistics.
    
    Get comprehensive statistics about clustering quality, size distribution,
    and reduction efficiency.
    
    Args:
        cluster_input: Path to ClusterSet JSON file
        
    Returns:
        Dictionary with detailed clustering analysis
    """
    
    try:
        # Load cluster set
        if not Path(cluster_input).exists():
            return {
                "error": f"Input file not found: {cluster_input}",
                "success": False
            }
        
        with open(cluster_input, 'r') as f:
            data = json.load(f)
        
        if "cluster_set" not in data:
            return {
                "error": "Invalid input file - missing cluster_set",
                "success": False
            }
        
        cluster_data = data["cluster_set"]
        
        # Extract clustering statistics
        clusters = cluster_data["clusters"]
        total_sequences = cluster_data["total_sequences"]
        total_clusters = cluster_data["total_clusters"]
        
        # Calculate detailed statistics
        cluster_sizes = [cluster["size"] for cluster in clusters]
        
        # Size distribution
        size_distribution = {}
        for size in cluster_sizes:
            size_key = str(size) if size <= 10 else "10+"
            size_distribution[size_key] = size_distribution.get(size_key, 0) + 1
        
        # Quality metrics
        singleton_count = sum(1 for size in cluster_sizes if size == 1)
        large_cluster_count = sum(1 for size in cluster_sizes if size >= 10)
        
        analysis = {
            "overview": {
                "total_input_sequences": total_sequences,
                "total_clusters": total_clusters,
                "reduction_ratio": 1 - (total_clusters / total_sequences) if total_sequences > 0 else 0,
                "compression_factor": f"{total_sequences}:{total_clusters}"
            },
            "cluster_size_stats": {
                "min": min(cluster_sizes) if cluster_sizes else 0,
                "max": max(cluster_sizes) if cluster_sizes else 0,
                "mean": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
                "median": sorted(cluster_sizes)[len(cluster_sizes) // 2] if cluster_sizes else 0
            },
            "size_distribution": size_distribution,
            "cluster_categories": {
                "singletons": singleton_count,
                "small_clusters_2_5": sum(1 for size in cluster_sizes if 2 <= size <= 5),
                "medium_clusters_6_9": sum(1 for size in cluster_sizes if 6 <= size <= 9),
                "large_clusters_10plus": large_cluster_count
            },
            "quality_metrics": {
                "singleton_ratio": singleton_count / total_clusters if total_clusters > 0 else 0,
                "large_cluster_ratio": large_cluster_count / total_clusters if total_clusters > 0 else 0,
                "effective_reduction": (total_sequences - singleton_count) / total_sequences if total_sequences > 0 else 0
            },
            "top_clusters": sorted(
                [(cluster["cluster_id"], cluster["size"]) for cluster in clusters],
                key=lambda x: x[1], reverse=True
            )[:10],
            "clustering_parameters": {
                "method": cluster_data.get("clustering_method", "unknown"),
                "identity_threshold": cluster_data.get("identity_threshold", "unknown"),
                "coverage_threshold": cluster_data.get("coverage_threshold", "unknown")
            }
        }
        
        return {
            "analysis": analysis,
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__,
            "success": False
        }


def _serialize_enums(obj):
    """Convert enum values to strings recursively."""
    if isinstance(obj, dict):
        return {k: _serialize_enums(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_enums(item) for item in obj]
    elif hasattr(obj, 'value'):  # Enum
        return obj.value
    else:
        return obj


# Create module initialization
def create_mcp_server():
    """Create and return the MCP server instance."""
    return mcp_server


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) > 1:
        mcp_server.run(sys.argv[1:])
    else:
        mcp_server.run()
