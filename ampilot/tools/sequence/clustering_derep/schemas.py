"""Data schemas for clustering and dereplication."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ClusteringMethod(Enum):
    """Clustering methods."""
    CD_HIT = "cd-hit"
    MMSEQS2 = "mmseqs2"
    USEARCH = "usearch"


class ClusteringMode(Enum):
    """Clustering modes."""
    IDENTITY = "identity"       # Based on sequence identity
    COVERAGE = "coverage"       # Based on coverage
    HYBRID = "hybrid"          # Identity + coverage


@dataclass
class ClusterMember:
    """Individual sequence in a cluster."""
    sequence_id: str
    sequence: str
    length: int
    is_representative: bool = False
    similarity_to_rep: Optional[float] = None
    coverage_to_rep: Optional[float] = None


@dataclass
class Cluster:
    """Sequence cluster."""
    cluster_id: str
    representative: ClusterMember
    members: List[ClusterMember]
    size: int
    avg_length: float
    identity_threshold: float
    coverage_threshold: Optional[float] = None
    
    def __post_init__(self):
        self.size = len(self.members) + 1  # +1 for representative


@dataclass
class ClusterSet:
    """Collection of sequence clusters."""
    clusters: List[Cluster]
    total_clusters: int
    total_sequences: int
    clustering_method: ClusteringMethod
    clustering_mode: ClusteringMode
    identity_threshold: float
    coverage_threshold: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.total_clusters = len(self.clusters)
        self.total_sequences = sum(cluster.size for cluster in self.clusters)


@dataclass
class ClusteringStats:
    """Statistics about clustering results."""
    total_input_sequences: int
    total_clusters: int
    largest_cluster_size: int
    smallest_cluster_size: int
    mean_cluster_size: float
    median_cluster_size: float
    singletons: int
    reduction_ratio: float


# Aliases for backward compatibility with tests
@dataclass
class ClusterMemberCompat:
    """Cluster member (for backward compatibility)."""
    sequence_id: str
    identity_to_representative: float
    coverage: float
    is_representative: bool

@dataclass
class ClusterCompat:
    """Cluster (for backward compatibility)."""
    cluster_id: str
    representative_id: str
    size: int
    members: List[ClusterMemberCompat]

@dataclass
class ClusterSetCompat:
    """Cluster set (for backward compatibility)."""
    clusters: List[ClusterCompat]
    total_clusters: int
    total_sequences: int
    representatives_count: int
    largest_cluster_size: int
    mean_cluster_size: float

@dataclass
class DedupBatch:
    """Deduplicated batch (for backward compatibility)."""
    representatives: List[str]
    total_representatives: int

# Use compatible versions for tests
ClusterMember = ClusterMemberCompat
Cluster = ClusterCompat  
ClusterSet = ClusterSetCompat
