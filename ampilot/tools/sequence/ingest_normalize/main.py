"""Main processing functions for sequence ingestion and normalization."""

from typing import List, Union, Dict, Any
from pathlib import Path
from collections import Counter

from .schemas import Sequence, SequenceBatch, QCReport, SequenceType, QualityFlag
from .parsers import FastaParser, SequenceDetector
from .quality_control import QualityController, SequenceDeduplicator


def process_sequences(
    input_data: Union[str, Path, List[str]],
    input_format: str = "fasta",
    min_length: int = 10,
    max_length: int = 10000,
    remove_duplicates: bool = True,
    low_complexity_threshold: float = 0.3,
    detect_sequence_type: bool = True
) -> tuple[SequenceBatch, QCReport]:
    """
    Process sequences with quality control and normalization.
    
    Args:
        input_data: File path, sequences as strings, or list of sequences
        input_format: Format of input data ("fasta", "string_list")
        min_length: Minimum sequence length
        max_length: Maximum sequence length  
        remove_duplicates: Whether to remove duplicate sequences
        low_complexity_threshold: Threshold for low complexity detection
        detect_sequence_type: Whether to auto-detect sequence type
        
    Returns:
        Tuple of (SequenceBatch, QCReport)
    """
    
    # Parse input sequences
    sequences = _parse_input(input_data, input_format)
    
    # Detect sequence types if requested
    if detect_sequence_type:
        for seq in sequences:
            seq.sequence_type = SequenceDetector.detect_type(seq.sequence)
    
    # Initialize quality controller
    qc = QualityController(
        min_length=min_length,
        max_length=max_length,
        low_complexity_threshold=low_complexity_threshold
    )
    
    # Initialize deduplicator
    dedup = SequenceDeduplicator(case_sensitive=False)
    
    # Find duplicates first (before removal)
    sequences = dedup.find_duplicates(sequences)
    
    # Apply quality control
    for seq in sequences:
        flags = qc.check_sequence(seq)
        seq.quality_flags.extend(flags)
    
    # Remove duplicates if requested
    if remove_duplicates:
        sequences = [seq for seq in sequences if QualityFlag.DUPLICATE not in seq.quality_flags]
    
    # Create batch and report
    batch = _create_sequence_batch(sequences)
    report = _create_qc_report(sequences)
    
    return batch, report


def _parse_input(input_data: Union[str, Path, List[str]], input_format: str) -> List[Sequence]:
    """Parse input data based on format."""
    if input_format == "fasta":
        if isinstance(input_data, (str, Path)):
            # File path
            return FastaParser.parse_file(input_data)
        else:
            # FASTA string
            return FastaParser.parse_string(str(input_data))
    
    elif input_format == "string_list":
        # List of sequences
        sequences = []
        for i, seq_str in enumerate(input_data):
            sequences.append(Sequence(
                id=f"seq_{i+1}",
                sequence=seq_str.upper()
            ))
        return sequences
    
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def _create_sequence_batch(sequences: List[Sequence]) -> SequenceBatch:
    """Create SequenceBatch from processed sequences."""
    valid_sequences = [seq for seq in sequences if QualityFlag.VALID in seq.quality_flags]
    
    metadata = {
        "sequence_types": _count_sequence_types(sequences),
        "average_length": sum(seq.length for seq in valid_sequences) / len(valid_sequences) if valid_sequences else 0
    }
    
    return SequenceBatch(
        sequences=sequences,
        total_count=len(sequences),
        valid_count=len(valid_sequences),
        metadata=metadata
    )


def _create_qc_report(sequences: List[Sequence]) -> QCReport:
    """Create QC report from processed sequences."""
    flag_counts = Counter()
    flag_details = {}
    
    for seq in sequences:
        for flag in seq.quality_flags:
            flag_counts[flag] += 1
            if flag not in flag_details:
                flag_details[flag] = []
            flag_details[flag].append(seq.id)
    
    return QCReport(
        total_sequences=len(sequences),
        valid_sequences=flag_counts.get(QualityFlag.VALID, 0),
        duplicate_count=flag_counts.get(QualityFlag.DUPLICATE, 0),
        too_short_count=flag_counts.get(QualityFlag.TOO_SHORT, 0),
        too_long_count=flag_counts.get(QualityFlag.TOO_LONG, 0),
        invalid_chars_count=flag_counts.get(QualityFlag.INVALID_CHARS, 0),
        low_complexity_count=flag_counts.get(QualityFlag.LOW_COMPLEXITY, 0),
        flag_details={flag.value: ids for flag, ids in flag_details.items()}
    )


def _count_sequence_types(sequences: List[Sequence]) -> Dict[str, int]:
    """Count sequences by type."""
    type_counts = Counter()
    for seq in sequences:
        if seq.sequence_type:
            type_counts[seq.sequence_type.value] += 1
    return dict(type_counts)


def process_sequences_simple(
    input_file: str,
    output_file: str,
    min_length: int = 10,
    remove_duplicates: bool = True
) -> Dict[str, Any]:
    """
    Simple process sequences function compatible with tests.
    
    Args:
        input_file: Path to input FASTA file
        output_file: Path to save processed sequences
        min_length: Minimum sequence length
        remove_duplicates: Whether to remove duplicate sequences
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Parse sequences
        sequences = FastaParser.parse_file(input_file)
        
        # Quality control
        qc = QualityController(min_length=min_length)
        qc_report = qc.analyze_batch(sequences)
        
        # Filter sequences based on QC
        valid_sequences = [seq for seq in sequences if len(seq.sequence) >= min_length]
        
        # Remove duplicates if requested
        if remove_duplicates:
            seen_sequences = set()
            unique_sequences = []
            for seq in valid_sequences:
                if seq.sequence not in seen_sequences:
                    unique_sequences.append(seq)
                    seen_sequences.add(seq.sequence)
            valid_sequences = unique_sequences
        
        # Create batch
        from .schemas import SequenceBatch
        sequence_batch = SequenceBatch(
            sequences=valid_sequences,
            total_count=len(valid_sequences),
            valid_count=len(valid_sequences),
            metadata={}
        )
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                for seq in valid_sequences:
                    f.write(f">{seq.id}\n{seq.sequence}\n")
        
        return {
            "status": "success",
            "sequence_batch": sequence_batch,
            "qc_report": qc_report,
            "output_file": output_file
        }
        
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": f"File not found: {str(e)}"
        }
    except ValueError as e:
        return {
            "status": "error", 
            "error": f"Invalid format: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Processing failed: {str(e)}"
        }
