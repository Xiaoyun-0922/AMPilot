"""FastMCP wrapper for homology search."""

import asyncio
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import asdict

from fastmcp import FastMCP

from .main import search_homologs, filter_homolog_results
from .schemas import SearchEngine, SearchType, HomologSet


# Create FastMCP server instance
mcp_server = FastMCP("Homology Search")


@mcp_server.tool()
async def search_sequence_homologs(
    query_input: str,
    database_path: str,
    search_engine: str = "blast",
    search_type: Optional[str] = None,
    evalue_threshold: float = 0.001,
    max_hits: int = 100,
    num_threads: int = 1,
    blast_bin_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for homologous sequences using BLAST, MMseqs2, or HMMER.
    
    This tool searches for sequences similar to your query sequences in a reference database.
    Useful for finding related proteins, evolutionary relationships, or functional annotations.
    
    Args:
        query_input: Path to SequenceBatch JSON file, or list of sequence IDs from previous ingest
        database_path: Path to search database (BLAST, MMseqs2, etc.)
        search_engine: Search engine - "blast", "mmseqs2", "hmmer" (default: "blast")
        search_type: Search type - "blastp", "blastn", "blastx", etc. (auto-detected if None)
        evalue_threshold: E-value threshold for significance (default: 0.001)
        max_hits: Maximum hits per query sequence (default: 100)
        num_threads: Number of CPU threads to use (default: 1)
        blast_bin_path: Path to BLAST+ binaries directory (optional)
        output_path: Path to save results as JSON (optional)
        
    Returns:
        Dictionary containing:
        - homolog_set: Search results with hits for each query
        - summary: Statistics about the search
        - output_file: Path to saved results (if output_path provided)
    """
    
    try:
        # Parse search engine
        try:
            engine = SearchEngine(search_engine.lower())
        except ValueError:
            return {
                "error": f"Unsupported search engine: {search_engine}. Supported: blast, mmseqs2, hmmer",
                "success": False
            }
        
        # Parse search type if provided
        search_type_enum = None
        if search_type:
            try:
                search_type_enum = SearchType(search_type.lower())
            except ValueError:
                return {
                    "error": f"Unsupported search type: {search_type}",
                    "success": False
                }
        
        # Validate database exists
        if not Path(database_path).exists():
            return {
                "error": f"Database not found: {database_path}",
                "success": False
            }
        
        # Run homology search
        homolog_set = await asyncio.get_event_loop().run_in_executor(
            None,
            search_homologs,
            query_input,
            database_path,
            engine,
            search_type_enum,
            evalue_threshold,
            max_hits,
            num_threads,
            blast_bin_path
        )
        
        # Convert to dictionary for JSON serialization
        homolog_dict = asdict(homolog_set)
        homolog_dict = _serialize_enums(homolog_dict)
        
        # Create summary statistics
        summary = {
            "total_queries": homolog_set.total_queries,
            "total_hits": homolog_set.total_hits,
            "queries_with_hits": sum(1 for result in homolog_set.search_results if result.hits),
            "avg_hits_per_query": homolog_set.total_hits / homolog_set.total_queries if homolog_set.total_queries > 0 else 0,
            "search_engine": homolog_set.search_engine.value,
            "search_type": homolog_set.search_type.value,
            "database": homolog_set.database_info.get("name", "unknown")
        }
        
        result = {
            "homolog_set": homolog_dict,
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
async def filter_homology_results(
    homolog_input: str,
    min_identity: float = 0.0,
    min_coverage: float = 0.0,
    max_evalue: float = 1.0,
    min_bit_score: float = 0.0,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter homology search results based on quality criteria.
    
    Apply filters to remove low-quality hits and retain only significant matches.
    
    Args:
        homolog_input: Path to HomologSet JSON file from previous search
        min_identity: Minimum identity percentage (0-100, default: 0)
        min_coverage: Minimum query coverage percentage (0-100, default: 0)
        max_evalue: Maximum E-value threshold (default: 1.0)
        min_bit_score: Minimum bit score (default: 0.0)
        output_path: Path to save filtered results (optional)
        
    Returns:
        Dictionary containing filtered homology results and statistics
    """
    
    try:
        # Load homolog set from file
        if not Path(homolog_input).exists():
            return {
                "error": f"Input file not found: {homolog_input}",
                "success": False
            }
        
        with open(homolog_input, 'r') as f:
            data = json.load(f)
        
        if "homolog_set" not in data:
            return {
                "error": "Invalid input file format - missing homolog_set",
                "success": False
            }
        
        # Reconstruct HomologSet object (simplified)
        homolog_data = data["homolog_set"]
        
        # Apply filtering
        # Note: For simplicity, we'll work with the dict representation
        # In a full implementation, you'd reconstruct the objects
        
        filtered_results = []
        total_hits_before = 0
        total_hits_after = 0
        
        for search_result in homolog_data["search_results"]:
            filtered_hits = []
            total_hits_before += len(search_result["hits"])
            
            for hit in search_result["hits"]:
                if (hit["identity"] >= min_identity and
                    hit["query_coverage"] >= min_coverage and
                    hit["evalue"] <= max_evalue and
                    hit["bit_score"] >= min_bit_score):
                    filtered_hits.append(hit)
            
            total_hits_after += len(filtered_hits)
            
            search_result["hits"] = filtered_hits
            search_result["total_hits"] = len(filtered_hits)
            filtered_results.append(search_result)
        
        # Update homolog set
        homolog_data["search_results"] = filtered_results
        homolog_data["total_hits"] = total_hits_after
        
        # Add filter metadata
        if "metadata" not in homolog_data:
            homolog_data["metadata"] = {}
        
        homolog_data["metadata"]["filtered"] = True
        homolog_data["metadata"]["filter_criteria"] = {
            "min_identity": min_identity,
            "min_coverage": min_coverage,
            "max_evalue": max_evalue,
            "min_bit_score": min_bit_score
        }
        
        # Create summary
        summary = {
            "hits_before_filtering": total_hits_before,
            "hits_after_filtering": total_hits_after,
            "hits_removed": total_hits_before - total_hits_after,
            "retention_rate": total_hits_after / total_hits_before if total_hits_before > 0 else 0,
            "queries_with_hits": sum(1 for result in filtered_results if result["hits"])
        }
        
        result = {
            "homolog_set": homolog_data,
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
async def check_database_info(database_path: str) -> Dict[str, Any]:
    """
    Check information about a sequence database.
    
    Args:
        database_path: Path to database files
        
    Returns:
        Dictionary with database information and status
    """
    
    try:
        db_path = Path(database_path)
        
        info = {
            "path": str(db_path),
            "name": db_path.name,
            "exists": db_path.exists(),
            "is_directory": db_path.is_dir() if db_path.exists() else False
        }
        
        if db_path.exists():
            # Check for BLAST database files
            blast_extensions = ['.phr', '.pin', '.psq', '.pdb', '.pot', '.ptf', '.pto']
            blast_files = []
            
            if db_path.is_dir():
                for ext in blast_extensions:
                    matches = list(db_path.glob(f"*{ext}"))
                    blast_files.extend(matches)
            else:
                # Check if files with same name but different extensions exist
                for ext in blast_extensions:
                    potential_file = db_path.parent / f"{db_path.name}{ext}"
                    if potential_file.exists():
                        blast_files.append(potential_file)
            
            info["blast_files_found"] = len(blast_files)
            info["blast_files"] = [str(f) for f in blast_files[:10]]  # First 10
            
            # Try to get database info using blastdbcmd
            try:
                import subprocess
                result = subprocess.run(
                    ["blastdbcmd", "-db", str(db_path), "-info"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    info["blast_info"] = result.stdout
                    # Parse for sequence count
                    import re
                    match = re.search(r'(\d+)\s+sequences', result.stdout)
                    if match:
                        info["sequence_count"] = int(match.group(1))
                else:
                    info["blast_info"] = f"Error: {result.stderr}"
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                info["blast_info"] = f"blastdbcmd not available: {e}"
        
        return info
        
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__
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
