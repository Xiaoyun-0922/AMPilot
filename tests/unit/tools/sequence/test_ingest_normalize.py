"""Unit tests for ingest_normalize module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules we want to test
from ampilot.tools.sequence.ingest_normalize.schemas import (
    SequenceRecord, SequenceBatch, QCReport
)
from ampilot.tools.sequence.ingest_normalize.parsers import FASTAParser
from ampilot.tools.sequence.ingest_normalize.quality_control import SequenceQC  
from ampilot.tools.sequence.ingest_normalize.main import process_sequences_simple as process_sequences


class TestSequenceRecord:
    """Test SequenceRecord data class."""
    
    def test_sequence_record_creation(self):
        """Test creating a sequence record."""
        record = SequenceRecord(
            id="test_seq",
            sequence="MKLLNVINFV",
            description="Test sequence"
        )
        assert record.id == "test_seq"
        assert record.sequence == "MKLLNVINFV"
        assert record.description == "Test sequence"
        assert record.length == 10

    def test_sequence_record_length_calculation(self):
        """Test automatic length calculation."""
        record = SequenceRecord(id="test", sequence="ACDEFGHIKLMN")
        assert record.length == 12


class TestFASTAParser:
    """Test FASTA file parsing."""
    
    def test_parse_valid_fasta(self):
        """Test parsing valid FASTA content."""
        fasta_content = """>seq1 description
MKLLNVINFV
>seq2
ACDEFGHIKL"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            f.flush()
            
            parser = FASTAParser()
            records = parser.parse_file(f.name)
            
            assert len(records) == 2
            assert records[0].id == "seq1"
            assert records[0].sequence == "MKLLNVINFV"
            assert records[0].description == "description"
            assert records[1].id == "seq2"
            assert records[1].sequence == "ACDEFGHIKL"
            
        Path(f.name).unlink()  # Clean up

    def test_parse_multiline_sequences(self):
        """Test parsing FASTA with multiline sequences."""
        fasta_content = """>seq1
MKLLNVINFV
FLMFVSSSAQ
>seq2
ACDEFG
HIKL"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            f.flush()
            
            parser = FASTAParser()
            records = parser.parse_file(f.name)
            
            assert len(records) == 2
            assert records[0].sequence == "MKLLNVINFVFLMFVSSSAQ"
            assert records[1].sequence == "ACDEFGHIKL"
            
        Path(f.name).unlink()

    def test_parse_empty_file(self):
        """Test parsing empty FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write("")
            f.flush()
            
            parser = FASTAParser()
            records = parser.parse_file(f.name)
            
            assert len(records) == 0
            
        Path(f.name).unlink()

    def test_parse_invalid_characters(self):
        """Test parsing FASTA with invalid characters."""
        fasta_content = """>seq1
MKLL123NVINFV"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            f.flush()
            
            parser = FASTAParser()
            with pytest.raises(ValueError, match="Invalid amino acid"):
                parser.parse_file(f.name)
                
        Path(f.name).unlink()


class TestSequenceQC:
    """Test sequence quality control."""
    
    def test_basic_qc_metrics(self):
        """Test basic QC metrics calculation."""
        records = [
            SequenceRecord(id="seq1", sequence="MKLLNVINFV", description=""),
            SequenceRecord(id="seq2", sequence="ACDEFGHIKL", description=""),
        ]
        
        qc = SequenceQC()
        report = qc.analyze_batch(records)
        
        assert report.total_sequences == 2
        assert report.valid_sequences == 2
        assert report.invalid_sequences == 0
        assert report.duplicate_sequences == 0
        assert report.mean_length == 10.0
        assert report.min_length == 10
        assert report.max_length == 10

    def test_duplicate_detection(self):
        """Test duplicate sequence detection."""
        records = [
            SequenceRecord(id="seq1", sequence="MKLLNVINFV", description=""),
            SequenceRecord(id="seq2", sequence="ACDEFGHIKL", description=""),
            SequenceRecord(id="seq3", sequence="MKLLNVINFV", description=""),  # duplicate
        ]
        
        qc = SequenceQC()
        report = qc.analyze_batch(records)
        
        assert report.total_sequences == 3
        assert report.duplicate_sequences == 1

    def test_low_complexity_detection(self):
        """Test low complexity sequence detection."""
        records = [
            SequenceRecord(id="seq1", sequence="AAAAAAAAAA", description=""),  # low complexity
            SequenceRecord(id="seq2", sequence="ACDEFGHIKL", description=""),
        ]
        
        qc = SequenceQC()
        report = qc.analyze_batch(records)
        
        # Should detect low complexity sequence
        assert any("low_complexity" in issue for issue in report.issues)

    def test_short_sequence_detection(self):
        """Test short sequence detection."""
        records = [
            SequenceRecord(id="seq1", sequence="MK", description=""),  # too short
            SequenceRecord(id="seq2", sequence="ACDEFGHIKL", description=""),
        ]
        
        qc = SequenceQC(min_length=5)
        report = qc.analyze_batch(records)
        
        assert any("too_short" in issue for issue in report.issues)


class TestProcessSequences:
    """Test main processing function."""
    
    @patch('ampilot.tools.sequence.ingest_normalize.main.FASTAParser')
    @patch('ampilot.tools.sequence.ingest_normalize.main.SequenceQC')
    def test_process_sequences_success(self, mock_qc_class, mock_parser_class):
        """Test successful sequence processing."""
        # Mock parser
        mock_parser = Mock()
        mock_records = [
            SequenceRecord(id="seq1", sequence="MKLLNVINFV", description=""),
            SequenceRecord(id="seq2", sequence="ACDEFGHIKL", description=""),
        ]
        mock_parser.parse_file.return_value = mock_records
        mock_parser_class.return_value = mock_parser
        
        # Mock QC
        mock_qc = Mock()
        mock_report = QCReport(
            total_sequences=2,
            valid_sequences=2,
            invalid_sequences=0,
            duplicate_sequences=0,
            mean_length=10.0,
            min_length=10,
            max_length=10,
            issues=[]
        )
        mock_qc.analyze_batch.return_value = mock_report
        mock_qc_class.return_value = mock_qc
        
        # Test processing
        result = process_sequences(
            input_file="test.fasta",
            output_file="output.fasta",
            min_length=5,
            remove_duplicates=True
        )
        
        assert result["status"] == "success"
        assert "sequence_batch" in result
        assert "qc_report" in result
        
        # Verify calls
        mock_parser.parse_file.assert_called_once_with("test.fasta")
        mock_qc.analyze_batch.assert_called_once()

    @patch('ampilot.tools.sequence.ingest_normalize.main.FASTAParser')
    def test_process_sequences_file_not_found(self, mock_parser_class):
        """Test processing with file not found error."""
        mock_parser = Mock()
        mock_parser.parse_file.side_effect = FileNotFoundError("File not found")
        mock_parser_class.return_value = mock_parser
        
        result = process_sequences(
            input_file="nonexistent.fasta",
            output_file="output.fasta"
        )
        
        assert result["status"] == "error"
        assert "File not found" in result["error"]

    @patch('ampilot.tools.sequence.ingest_normalize.main.FASTAParser')
    def test_process_sequences_invalid_format(self, mock_parser_class):
        """Test processing with invalid file format."""
        mock_parser = Mock()
        mock_parser.parse_file.side_effect = ValueError("Invalid format")
        mock_parser_class.return_value = mock_parser
        
        result = process_sequences(
            input_file="invalid.fasta",
            output_file="output.fasta"
        )
        
        assert result["status"] == "error"
        assert "Invalid format" in result["error"]
