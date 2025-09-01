"""Data schemas for sequence ingestion and normalization."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SequenceType(Enum):
    """Sequence type enumeration."""
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"


class QualityFlag(Enum):
    """Quality control flags."""
    VALID = "valid"
    DUPLICATE = "duplicate"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    INVALID_CHARS = "invalid_chars"
    LOW_COMPLEXITY = "low_complexity"


@dataclass
class Sequence:
    """Individual sequence data."""
    id: str
    sequence: str
    description: Optional[str] = None
    length: Optional[int] = None
    sequence_type: Optional[SequenceType] = None
    quality_flags: List[QualityFlag] = None
    
    def __post_init__(self):
        if self.length is None:
            self.length = len(self.sequence)
        if self.quality_flags is None:
            self.quality_flags = []


@dataclass
class SequenceBatch:
    """Batch of processed sequences."""
    sequences: List[Sequence]
    total_count: int
    valid_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QCMetrics:
    """Quality control metrics (for backward compatibility)."""
    mean_length: float
    min_length: int
    max_length: int
    gc_content: Optional[float] = None


@dataclass
class QCReportCompat:
    """QC Report compatible with tests."""
    total_sequences: int
    valid_sequences: int
    invalid_sequences: int
    duplicate_sequences: int
    mean_length: float
    min_length: int
    max_length: int
    issues: List[str]


# Aliases for backward compatibility with tests
SequenceRecord = Sequence

# Use the compatible version as the main QCReport for tests
QCReport = QCReportCompat
