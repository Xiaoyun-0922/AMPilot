"""Data schemas for homology search."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SearchEngine(Enum):
    """Supported search engines."""
    BLAST = "blast"
    MMSEQS2 = "mmseqs2"
    HMMER = "hmmer"


class SearchType(Enum):
    """Types of homology searches."""
    BLASTP = "blastp"    # protein vs protein
    BLASTN = "blastn"    # nucleotide vs nucleotide
    BLASTX = "blastx"    # nucleotide vs protein
    TBLASTN = "tblastn"  # protein vs nucleotide
    TBLASTX = "tblastx"  # nucleotide vs nucleotide (translated)
    MMSEQS_SEARCH = "mmseqs_search"
    HMMSEARCH = "hmmsearch"


@dataclass
class SearchHit:
    """Individual homology search hit."""
    target_id: str
    target_description: Optional[str]
    query_id: str
    evalue: float
    bit_score: float
    identity: float
    similarity: Optional[float]
    query_coverage: float
    target_coverage: float
    alignment_length: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    query_sequence: Optional[str] = None
    target_sequence: Optional[str] = None
    alignment_string: Optional[str] = None


@dataclass
class SearchResult:
    """Search results for a single query."""
    query_id: str
    query_length: int
    hits: List[SearchHit]
    total_hits: int
    search_params: Dict[str, Any]
    execution_time: Optional[float] = None


@dataclass
class HomologSet:
    """Collection of homology search results."""
    search_results: List[SearchResult]
    total_queries: int
    total_hits: int
    search_engine: SearchEngine
    search_type: SearchType
    database_info: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DatabaseInfo:
    """Information about search database."""
    name: str
    path: str
    type: str  # "protein", "nucleotide", "profile"
    size: int  # number of sequences
    date_created: Optional[str] = None
    description: Optional[str] = None


# Aliases for backward compatibility with tests
@dataclass
class HomologyHit:
    """Individual homology hit (for backward compatibility)."""
    query_id: str
    subject_id: str
    identity: float
    coverage: float
    e_value: float
    bit_score: float
    alignment_length: int
    query_start: int
    query_end: int
    subject_start: int
    subject_end: int
    subject_sequence: str

@dataclass
class HomologyResult:
    """Homology search result (for backward compatibility)."""
    query_id: str
    hits: List[HomologyHit]
    total_hits: int

@dataclass 
class HomologSet:
    """Set of homology search results (for backward compatibility)."""
    query_results: List[HomologyResult]
    total_queries: int
    total_hits: int
