"""Sequence parsers for different formats."""

import re
from typing import List, Iterator, Union, TextIO
from pathlib import Path
from .schemas import Sequence, SequenceType


class FastaParser:
    """FASTA format parser."""
    
    @staticmethod
    def parse_file(file_path: Union[str, Path]) -> List[Sequence]:
        """Parse FASTA file and return list of sequences."""
        sequences = []
        with open(file_path, 'r') as f:
            sequences.extend(FastaParser.parse_stream(f))
        return sequences
    
    @staticmethod
    def parse_string(fasta_string: str) -> List[Sequence]:
        """Parse FASTA string and return list of sequences."""
        lines = fasta_string.strip().split('\n')
        return FastaParser._parse_lines(lines)
    
    @staticmethod
    def parse_stream(stream: TextIO) -> Iterator[Sequence]:
        """Parse FASTA from stream."""
        lines = [line.strip() for line in stream]
        yield from FastaParser._parse_lines(lines)
    
    @staticmethod
    def _parse_lines(lines: List[str]) -> Iterator[Sequence]:
        """Parse FASTA lines and yield sequences."""
        current_id = None
        current_desc = None
        current_seq = []
        
        for line in lines:
            if not line:
                continue
                
            if line.startswith('>'):
                # Process previous sequence if exists
                if current_id is not None and current_seq:
                    yield Sequence(
                        id=current_id,
                        sequence=''.join(current_seq),
                        description=current_desc
                    )
                
                # Start new sequence
                header = line[1:]  # Remove '>'
                parts = header.split(None, 1)  # Split on first whitespace
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else None
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        # Process last sequence
        if current_id is not None and current_seq:
            yield Sequence(
                id=current_id,
                sequence=''.join(current_seq),
                description=current_desc
            )


class SequenceDetector:
    """Detect sequence type (protein, DNA, RNA)."""
    
    DNA_CHARS = set('ATCG')
    RNA_CHARS = set('AUCG')
    PROTEIN_CHARS = set('ACDEFGHIKLMNPQRSTVWY')
    AMBIGUOUS_CHARS = set('NXBZ-')
    
    @classmethod
    def detect_type(cls, sequence: str) -> SequenceType:
        """Detect sequence type based on character composition."""
        seq_clean = sequence.upper().replace('-', '').replace('N', '').replace('X', '')
        
        if not seq_clean:
            return SequenceType.PROTEIN  # Default fallback
        
        chars = set(seq_clean)
        
        # Check for DNA
        if chars.issubset(cls.DNA_CHARS | cls.AMBIGUOUS_CHARS):
            return SequenceType.DNA
        
        # Check for RNA
        if chars.issubset(cls.RNA_CHARS | cls.AMBIGUOUS_CHARS):
            return SequenceType.RNA
        
        # Check for protein
        if chars.issubset(cls.PROTEIN_CHARS | cls.AMBIGUOUS_CHARS):
            return SequenceType.PROTEIN
        
        # Default to protein
        return SequenceType.PROTEIN

# Alias for backward compatibility with tests
FASTAParser = FastaParser
