"""Quality control functions for sequences."""

import re
from typing import List, Set
from collections import Counter
from .schemas import Sequence, QualityFlag, SequenceType, QCReport


class QualityController:
    """Quality control for sequences."""
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 10000,
                 low_complexity_threshold: float = 0.3):
        self.min_length = min_length
        self.max_length = max_length
        self.low_complexity_threshold = low_complexity_threshold
    
    def check_sequence(self, sequence: Sequence) -> List[QualityFlag]:
        """Check a single sequence and return quality flags."""
        flags = []
        
        # Check length
        if sequence.length < self.min_length:
            flags.append(QualityFlag.TOO_SHORT)
        elif sequence.length > self.max_length:
            flags.append(QualityFlag.TOO_LONG)
        
        # Check for invalid characters
        if self._has_invalid_chars(sequence):
            flags.append(QualityFlag.INVALID_CHARS)
        
        # Check complexity
        if self._is_low_complexity(sequence):
            flags.append(QualityFlag.LOW_COMPLEXITY)
        
        # If no issues found, mark as valid
        if not flags:
            flags.append(QualityFlag.VALID)
        
        return flags
    
    def _has_invalid_chars(self, sequence: Sequence) -> bool:
        """Check if sequence contains invalid characters."""
        valid_chars = self._get_valid_chars(sequence.sequence_type)
        seq_chars = set(sequence.sequence.upper())
        invalid_chars = seq_chars - valid_chars
        return len(invalid_chars) > 0
    
    def _get_valid_chars(self, seq_type: SequenceType) -> Set[str]:
        """Get valid characters for sequence type."""
        if seq_type == SequenceType.DNA:
            return set('ATCGNBDHKMNRSTVWY-')  # Include ambiguous codes
        elif seq_type == SequenceType.RNA:
            return set('AUCGNBDHKMNRSTVWY-')  # Include ambiguous codes
        else:  # PROTEIN
            return set('ACDEFGHIKLMNPQRSTVWYXBZJOU*-')  # Include stop codon and ambiguous
    
    def _is_low_complexity(self, sequence: Sequence) -> bool:
        """Check if sequence has low complexity."""
        seq = sequence.sequence.upper().replace('-', '')
        if len(seq) < 4:
            return False
        
        # Calculate entropy
        char_counts = Counter(seq)
        total_chars = len(seq)
        
        # Shannon entropy
        entropy = 0
        for count in char_counts.values():
            p = count / total_chars
            entropy -= p * (p and p.bit_length() - 1)  # log2 approximation
        
        # Normalize by max possible entropy
        max_entropy = (len(char_counts) and 
                       len(char_counts).bit_length() - 1) or 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy < self.low_complexity_threshold
    
    def analyze_batch(self, sequences: List[Sequence]) -> 'QCReport':
        """Analyze a batch of sequences and generate QC report."""
        from .schemas import QCReport
        
        total_sequences = len(sequences)
        valid_sequences = 0
        invalid_sequences = 0
        duplicate_sequences = 0
        lengths = []
        issues = []
        
        seen_sequences = set()
        
        for seq in sequences:
            length = len(seq.sequence)
            lengths.append(length)
            
            # Check duplicates
            if seq.sequence in seen_sequences:
                duplicate_sequences += 1
                issues.append(f"Duplicate sequence: {seq.id}")
            else:
                seen_sequences.add(seq.sequence)
            
            # Check quality flags
            flags = self.check_sequence(seq)
            has_issues = False
            
            for flag in flags:
                if flag == QualityFlag.TOO_SHORT:
                    invalid_sequences += 1
                    issues.append(f"Sequence too short: {seq.id}")
                    has_issues = True
                elif flag == QualityFlag.TOO_LONG:
                    invalid_sequences += 1
                    issues.append(f"Sequence too long: {seq.id}")
                    has_issues = True
                elif flag == QualityFlag.LOW_COMPLEXITY:
                    issues.append(f"Low complexity sequence: {seq.id}")
                elif flag == QualityFlag.INVALID_CHARS:
                    invalid_sequences += 1
                    issues.append(f"Invalid characters in sequence: {seq.id}")
                    has_issues = True
            
            if not has_issues:
                valid_sequences += 1
        
        # Calculate statistics
        mean_length = sum(lengths) / len(lengths) if lengths else 0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0
        
        return QCReport(
            total_sequences=total_sequences,
            valid_sequences=valid_sequences,
            invalid_sequences=invalid_sequences,
            duplicate_sequences=duplicate_sequences,
            mean_length=mean_length,
            min_length=min_length,
            max_length=max_length,
            issues=issues
        )


class SequenceDeduplicator:
    """Remove duplicate sequences."""
    
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
    
    def find_duplicates(self, sequences: List[Sequence]) -> List[Sequence]:
        """Find and mark duplicate sequences."""
        seen_sequences = {}
        result = []
        
        for seq in sequences:
            seq_key = seq.sequence if self.case_sensitive else seq.sequence.upper()
            
            if seq_key in seen_sequences:
                # Mark as duplicate
                seq.quality_flags.append(QualityFlag.DUPLICATE)
            else:
                seen_sequences[seq_key] = seq.id
            
            result.append(seq)
        
        return result
    
    def remove_duplicates(self, sequences: List[Sequence]) -> List[Sequence]:
        """Remove duplicate sequences, keeping first occurrence."""
        seen_sequences = set()
        result = []
        
        for seq in sequences:
            seq_key = seq.sequence if self.case_sensitive else seq.sequence.upper()
            
            if seq_key not in seen_sequences:
                seen_sequences.add(seq_key)
                result.append(seq)
        
        return result

# Alias for backward compatibility with tests
SequenceQC = QualityController
