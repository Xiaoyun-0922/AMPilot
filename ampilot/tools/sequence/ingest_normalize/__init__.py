"""FastMCP wrapper for sequence ingestion and normalization."""

import asyncio
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import tempfile
from dataclasses import asdict

from fastmcp import FastMCP

from .main import process_sequences
from .schemas import SequenceBatch, QCReport


# Create FastMCP server instance
mcp_server = FastMCP("Sequence Ingest & Normalize")


@mcp_server.tool()
async def ingest_normalize_sequences(
    input_data: str,
    input_format: str = "fasta",
    min_length: int = 10,
    max_length: int = 10000,
    remove_duplicates: bool = True,
    low_complexity_threshold: float = 0.3,
    detect_sequence_type: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest and normalize protein/DNA/RNA sequences with quality control.
    
    This tool loads sequences from various formats, performs quality control checks,
    removes duplicates, and generates a clean sequence batch for downstream analysis.
    
    Args:
        input_data: FASTA file path, FASTA string, or JSON list of sequences
        input_format: Input format - "fasta" (file or string) or "string_list" 
        min_length: Minimum sequence length (default: 10)
        max_length: Maximum sequence length (default: 10000)
        remove_duplicates: Remove duplicate sequences (default: True)
        low_complexity_threshold: Low complexity detection threshold 0-1 (default: 0.3)
        detect_sequence_type: Auto-detect DNA/RNA/protein (default: True)
        output_path: Optional path to save results as JSON
        
    Returns:
        Dictionary containing:
        - batch: Processed sequence batch with metadata
        - qc_report: Quality control report with statistics
        - output_file: Path to saved results (if output_path provided)
    """
    
    try:
        # Parse input data based on format
        if input_format == "string_list":
            # Parse JSON list
            sequences_list = json.loads(input_data) if isinstance(input_data, str) else input_data
            processed_input = sequences_list
        else:
            # FASTA format - could be file path or string
            if Path(input_data).exists():
                # File path
                processed_input = Path(input_data)
            else:
                # FASTA string content
                processed_input = input_data
        
        # Process sequences
        batch, qc_report = await asyncio.get_event_loop().run_in_executor(
            None,
            process_sequences,
            processed_input,
            input_format,
            min_length,
            max_length,
            remove_duplicates,
            low_complexity_threshold,
            detect_sequence_type
        )
        
        # Convert to dictionaries for JSON serialization
        batch_dict = asdict(batch)
        report_dict = asdict(qc_report)
        
        # Convert enum values to strings
        batch_dict = _serialize_enums(batch_dict)
        report_dict = _serialize_enums(report_dict)
        
        result = {
            "batch": batch_dict,
            "qc_report": report_dict,
            "summary": {
                "total_sequences": batch.total_count,
                "valid_sequences": batch.valid_count,
                "success_rate": batch.valid_count / batch.total_count if batch.total_count > 0 else 0
            }
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
async def validate_fasta_format(fasta_content: str) -> Dict[str, Any]:
    """
    Validate FASTA format and get basic statistics without full processing.
    
    Args:
        fasta_content: FASTA format string to validate
        
    Returns:
        Dictionary with validation results and basic statistics
    """
    
    try:
        from .parsers import FastaParser
        
        # Parse FASTA
        sequences = await asyncio.get_event_loop().run_in_executor(
            None,
            FastaParser.parse_string,
            fasta_content
        )
        
        if not sequences:
            return {
                "valid": False,
                "error": "No sequences found in input",
                "count": 0
            }
        
        # Basic statistics
        lengths = [len(seq.sequence) for seq in sequences]
        
        return {
            "valid": True,
            "count": len(sequences),
            "sequence_ids": [seq.id for seq in sequences[:10]],  # First 10 IDs
            "length_stats": {
                "min": min(lengths),
                "max": max(lengths),
                "average": sum(lengths) / len(lengths)
            },
            "sample_sequences": [
                {"id": seq.id, "length": len(seq.sequence), "preview": seq.sequence[:50] + "..." if len(seq.sequence) > 50 else seq.sequence}
                for seq in sequences[:3]  # First 3 sequences
            ]
        }
        
    except Exception as e:
        return {
            "valid": False,
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


# Create module __init__.py
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
