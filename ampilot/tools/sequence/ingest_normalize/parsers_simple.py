"""Simple FASTA parser."""

from typing import List
from .schemas import SequenceRecord


class FASTAParser:
    """Simple FASTA file parser."""
    
    def parse_file(self, file_path: str) -> List[SequenceRecord]:
        """Parse FASTA file and return sequence records."""
        sequences = []
        
        with open(file_path, 'r') as f:
            current_id = None
            current_description = ""
            current_sequence = ""
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_id is not None:
                        self._validate_sequence(current_sequence)
                        sequences.append(SequenceRecord(
                            id=current_id,
                            sequence=current_sequence,
                            description=current_description
                        ))
                    
                    # Parse new header
                    header_parts = line[1:].split(' ', 1)
                    current_id = header_parts[0]
                    current_description = header_parts[1] if len(header_parts) > 1 else ""
                    current_sequence = ""
                else:
                    current_sequence += line
            
            # Save last sequence
            if current_id is not None:
                self._validate_sequence(current_sequence)
                sequences.append(SequenceRecord(
                    id=current_id,
                    sequence=current_sequence,
                    description=current_description
                ))
        
        return sequences
    
    def _validate_sequence(self, sequence: str) -> None:
        """Validate sequence contains only valid amino acid characters."""
        valid_chars = set('ACDEFGHIKLMNPQRSTVWYX')
        for char in sequence.upper():
            if char not in valid_chars:
                raise ValueError(f"Invalid amino acid character: {char}")
