"""Common test fixtures for sequence analysis."""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_fasta_content():
    """Sample FASTA content for testing."""
    return """>sequence1
MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
KLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
>sequence2 description here
ACDEFGHIKLMNPQRSTVWYJKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHI
>sequence3|with|pipes
GGGGLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEY
SSGGKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFE
>duplicate_sequence
MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
KLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
>short_seq
MKLL
>sequence_with_X
MKXLNVINFVFLMFVSSSXQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG"""

@pytest.fixture
def sample_fasta_file(temp_dir, sample_fasta_content):
    """Create a sample FASTA file."""
    fasta_file = temp_dir / "sample.fasta"
    with open(fasta_file, 'w') as f:
        f.write(sample_fasta_content)
    return str(fasta_file)

@pytest.fixture
def invalid_fasta_content():
    """Invalid FASTA content for testing error handling."""
    return """>sequence1
MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
sequence2_no_header
ACDEFGHIKLMNPQRSTVWYJKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHI
>sequence3_with_invalid_chars
MKLLNVINFVFLMFVSSSAQAM123VDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEY"""

@pytest.fixture
def sample_protein_sequences():
    """Sample protein sequences as list."""
    return [
        {"id": "seq1", "sequence": "MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG"},
        {"id": "seq2", "sequence": "ACDEFGHIKLMNPQRSTVWYJKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHI"},
        {"id": "seq3", "sequence": "GGGGLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEY"},
    ]

@pytest.fixture
def mcp_test_config():
    """Configuration for MCP testing."""
    return {
        "server_name": "test_server",
        "version": "1.0.0",
        "timeout": 30,
        "max_file_size": 1000000,  # 1MB
    }

@pytest.fixture
def blast_test_db_path(temp_dir):
    """Mock BLAST database path."""
    db_path = temp_dir / "test_db"
    # Create mock database files
    for ext in ['.phr', '.pin', '.psq']:
        (db_path.parent / f"{db_path.name}{ext}").touch()
    return str(db_path)

@pytest.fixture
def cdhit_test_config():
    """CD-HIT test configuration."""
    return {
        "identity_threshold": 0.9,
        "coverage_threshold": 0.8,
        "word_length": 5,
        "memory_limit": 1000,  # MB
    }
