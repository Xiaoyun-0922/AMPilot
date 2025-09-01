"""Main processing functions for homology search."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

from ..ingest_normalize.schemas import Sequence, SequenceBatch
from .schemas import HomologSet, SearchResult, SearchEngine, SearchType, DatabaseInfo
from .blast_engine import BlastSearchEngine


def search_homologs(
    query_sequences: Union[List[Sequence], SequenceBatch, str, Path],
    database_path: str,
    search_engine: SearchEngine = SearchEngine.BLAST,
    search_type: Optional[SearchType] = None,
    evalue_threshold: float = 0.001,
    max_hits: int = 100,
    num_threads: int = 1,
    blast_bin_path: Optional[str] = None
) -> HomologSet:
    """
    Search for homologous sequences.
    
    Args:
        query_sequences: Input sequences (various formats supported)
        database_path: Path to search database
        search_engine: Search engine to use
        search_type: Type of search (auto-detected if None)
        evalue_threshold: E-value threshold for hits
        max_hits: Maximum number of hits per query
        num_threads: Number of threads to use
        blast_bin_path: Path to BLAST binaries
        
    Returns:
        HomologSet with search results
    """
    
    # Parse and validate input sequences
    sequences = _parse_query_input(query_sequences)
    
    if not sequences:
        raise ValueError("No valid query sequences provided")
    
    # Auto-detect search type if not provided
    if search_type is None:
        search_type = _detect_search_type(sequences)
    
    # Initialize search engine
    if search_engine == SearchEngine.BLAST:
        engine = BlastSearchEngine(blast_bin_path)
        search_results = engine.search(
            query_sequences=sequences,
            database_path=database_path,
            search_type=search_type,
            evalue_threshold=evalue_threshold,
            max_hits=max_hits,
            num_threads=num_threads
        )
    else:
        raise NotImplementedError(f"Search engine {search_engine} not yet implemented")
    
    # Get database info
    db_info = _get_database_info(database_path, search_engine)
    
    # Calculate statistics
    total_hits = sum(len(result.hits) for result in search_results)
    
    # Create HomologSet
    homolog_set = HomologSet(
        search_results=search_results,
        total_queries=len(sequences),
        total_hits=total_hits,
        search_engine=search_engine,
        search_type=search_type,
        database_info=db_info,
        metadata={
            "evalue_threshold": evalue_threshold,
            "max_hits_per_query": max_hits,
            "avg_hits_per_query": total_hits / len(sequences) if sequences else 0
        }
    )
    
    return homolog_set


def _parse_query_input(query_input: Union[List[Sequence], SequenceBatch, str, Path]) -> List[Sequence]:
    """Parse various input formats to List[Sequence]."""
    
    if isinstance(query_input, list) and all(isinstance(seq, Sequence) for seq in query_input):
        return query_input
    
    elif isinstance(query_input, SequenceBatch):
        # Extract valid sequences from batch
        return [seq for seq in query_input.sequences 
                if any(flag.value == "valid" for flag in seq.quality_flags)]
    
    elif isinstance(query_input, (str, Path)):
        # Load from JSON file (assume it's a saved SequenceBatch)
        with open(query_input, 'r') as f:
            data = json.load(f)
        
        if "batch" in data:
            # Extract sequences from saved batch
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
        else:
            raise ValueError("Invalid input file format")
    
    else:
        raise ValueError(f"Unsupported query input type: {type(query_input)}")


def _detect_search_type(sequences: List[Sequence]) -> SearchType:
    """Auto-detect appropriate search type based on sequence content."""
    
    # Check first sequence to determine type
    if not sequences:
        return SearchType.BLASTP  # Default
    
    first_seq = sequences[0]
    
    # Simple heuristic: check for common protein vs nucleotide characters
    seq_upper = first_seq.sequence.upper()
    nucleotide_chars = set('ATCGUN')
    protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
    
    seq_chars = set(seq_upper)
    
    # If sequence contains only nucleotide characters (plus ambiguous)
    if seq_chars.issubset(nucleotide_chars | set('BDHKMNRSTVWY-')):
        return SearchType.BLASTN
    
    # If sequence contains protein-specific characters
    elif seq_chars & (protein_chars - nucleotide_chars):
        return SearchType.BLASTP
    
    # Default to protein
    return SearchType.BLASTP


def _get_database_info(database_path: str, search_engine: SearchEngine) -> Dict[str, Any]:
    """Get information about the search database."""
    
    db_path = Path(database_path)
    
    # Basic info
    info = {
        "name": db_path.name,
        "path": str(db_path),
        "exists": db_path.exists()
    }
    
    if search_engine == SearchEngine.BLAST:
        # Try to get BLAST database info
        try:
            import subprocess
            result = subprocess.run(
                ["blastdbcmd", "-db", database_path, "-info"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                # Parse blastdbcmd output for sequence count, etc.
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "sequences" in line.lower():
                        # Extract number of sequences
                        import re
                        match = re.search(r'(\d+)\s+sequences', line)
                        if match:
                            info["size"] = int(match.group(1))
                        break
            else:
                info["size"] = "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            info["size"] = "unknown"
    
    return info


def filter_homolog_results(
    homolog_set: HomologSet,
    min_identity: float = 0.0,
    min_coverage: float = 0.0,
    max_evalue: float = 1.0,
    min_bit_score: float = 0.0
) -> HomologSet:
    """
    Filter homology search results based on various criteria.
    
    Args:
        homolog_set: Input HomologSet
        min_identity: Minimum identity percentage
        min_coverage: Minimum query coverage
        max_evalue: Maximum E-value
        min_bit_score: Minimum bit score
        
    Returns:
        Filtered HomologSet
    """
    
    filtered_results = []
    
    for search_result in homolog_set.search_results:
        filtered_hits = []
        
        for hit in search_result.hits:
            if (hit.identity >= min_identity and
                hit.query_coverage >= min_coverage and
                hit.evalue <= max_evalue and
                hit.bit_score >= min_bit_score):
                filtered_hits.append(hit)
        
        # Create new SearchResult with filtered hits
        filtered_result = SearchResult(
            query_id=search_result.query_id,
            query_length=search_result.query_length,
            hits=filtered_hits,
            total_hits=len(filtered_hits),
            search_params=search_result.search_params,
            execution_time=search_result.execution_time
        )
        filtered_results.append(filtered_result)
    
    # Update metadata
    total_hits = sum(len(result.hits) for result in filtered_results)
    
    filtered_set = HomologSet(
        search_results=filtered_results,
        total_queries=len(filtered_results),
        total_hits=total_hits,
        search_engine=homolog_set.search_engine,
        search_type=homolog_set.search_type,
        database_info=homolog_set.database_info,
        metadata={
            **homolog_set.metadata,
            "filtered": True,
            "filter_criteria": {
                "min_identity": min_identity,
                "min_coverage": min_coverage,
                "max_evalue": max_evalue,
                "min_bit_score": min_bit_score
            }
        }
    )
    
    return filtered_set
