"""Unit tests for homology_search module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ampilot.tools.sequence.homology_search.schemas import (
    HomologyHit, HomologyResult, HomologSet
)
from ampilot.tools.sequence.homology_search.blast_engine import BLASTPEngine
from ampilot.tools.sequence.homology_search.main import search_homologs


class TestHomologyHit:
    """Test HomologyHit data class."""
    
    def test_homology_hit_creation(self):
        """Test creating a homology hit."""
        hit = HomologyHit(
            query_id="query1",
            subject_id="subject1", 
            identity=85.5,
            coverage=90.0,
            e_value=1e-50,
            bit_score=150.0,
            alignment_length=100,
            query_start=1,
            query_end=100,
            subject_start=10,
            subject_end=109,
            subject_sequence="MKLLNVINFV"
        )
        
        assert hit.query_id == "query1"
        assert hit.identity == 85.5
        assert hit.coverage == 90.0
        assert hit.e_value == 1e-50


class TestBLASTPEngine:
    """Test BLAST+ engine wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = BLASTPEngine()

    @patch('subprocess.run')
    def test_run_blast_success(self, mock_run):
        """Test successful BLAST run."""
        # Mock BLAST output
        mock_blast_output = """query1\tsubject1\t85.5\t100\t0\t0\t1\t100\t10\t109\t1e-50\t150.0
query1\tsubject2\t80.0\t95\t5\t2\t1\t95\t15\t110\t1e-45\t140.0"""
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = mock_blast_output
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_file:
            query_file.write(">query1\nMKLLNVINFV")
            query_file.flush()
            
            engine = BLASTPEngine()
            results = engine.run_blast(
                query_file=query_file.name,
                database="/path/to/db",
                e_value=1e-5,
                max_hits=10
            )
            
            assert len(results) == 2
            assert results[0].query_id == "query1"
            assert results[0].subject_id == "subject1"
            assert results[0].identity == 85.5
            assert results[0].e_value == 1e-50
            
        Path(query_file.name).unlink()

    @patch('subprocess.run')
    def test_run_blast_failure(self, mock_run):
        """Test BLAST run failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Database not found"
        mock_run.return_value = mock_result
        
        engine = BLASTPEngine()
        with pytest.raises(RuntimeError, match="BLAST failed"):
            engine.run_blast(
                query_file="query.fasta",
                database="/nonexistent/db",
                e_value=1e-5,
                max_hits=10
            )

    @patch('subprocess.run')
    def test_run_blast_no_hits(self, mock_run):
        """Test BLAST run with no hits."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        engine = BLASTPEngine()
        results = engine.run_blast(
            query_file="query.fasta",
            database="/path/to/db",
            e_value=1e-5,
            max_hits=10
        )
        
        assert len(results) == 0

    def test_parse_blast_output(self):
        """Test parsing BLAST output."""
        blast_output = """query1\tsubject1\t85.5\t100\t0\t0\t1\t100\t10\t109\t1e-50\t150.0
query1\tsubject2\t80.0\t95\t5\t2\t1\t95\t15\t110\t1e-45\t140.0"""
        
        engine = BLASTPEngine()
        results = engine._parse_blast_output(blast_output)
        
        assert len(results) == 2
        
        hit1 = results[0]
        assert hit1.query_id == "query1"
        assert hit1.subject_id == "subject1"
        assert hit1.identity == 85.5
        assert hit1.alignment_length == 100
        assert hit1.e_value == 1e-50
        
        hit2 = results[1]
        assert hit2.query_id == "query1"
        assert hit2.subject_id == "subject2"
        assert hit2.identity == 80.0

    def test_filter_hits_by_criteria(self):
        """Test filtering hits by various criteria."""
        hits = [
            HomologyHit(
                query_id="q1", subject_id="s1", identity=90.0, coverage=95.0,
                e_value=1e-60, bit_score=200.0, alignment_length=100,
                query_start=1, query_end=100, subject_start=1, subject_end=100,
                subject_sequence="SEQ1"
            ),
            HomologyHit(
                query_id="q1", subject_id="s2", identity=70.0, coverage=80.0,
                e_value=1e-30, bit_score=120.0, alignment_length=80,
                query_start=1, query_end=80, subject_start=1, subject_end=80,
                subject_sequence="SEQ2"
            ),
            HomologyHit(
                query_id="q1", subject_id="s3", identity=95.0, coverage=60.0,
                e_value=1e-70, bit_score=220.0, alignment_length=60,
                query_start=1, query_end=60, subject_start=1, subject_end=60,
                subject_sequence="SEQ3"
            )
        ]
        
        engine = BLASTPEngine()
        
        # Filter by identity
        filtered = engine._filter_hits(hits, min_identity=80.0)
        assert len(filtered) == 2
        assert all(hit.identity >= 80.0 for hit in filtered)
        
        # Filter by coverage
        filtered = engine._filter_hits(hits, min_coverage=85.0)
        assert len(filtered) == 1
        assert filtered[0].coverage >= 85.0
        
        # Filter by e-value
        filtered = engine._filter_hits(hits, max_e_value=1e-50)
        assert len(filtered) == 2
        assert all(hit.e_value <= 1e-50 for hit in filtered)


class TestSearchHomologs:
    """Test main homology search function."""
    
    @patch('ampilot.tools.sequence.homology_search.main.BLASTPEngine')
    def test_search_homologs_success(self, mock_engine_class):
        """Test successful homology search."""
        # Mock BLAST engine
        mock_engine = Mock()
        mock_hits = [
            HomologyHit(
                query_id="query1", subject_id="subject1", identity=85.5, coverage=90.0,
                e_value=1e-50, bit_score=150.0, alignment_length=100,
                query_start=1, query_end=100, subject_start=10, subject_end=109,
                subject_sequence="MKLLNVINFV"
            )
        ]
        mock_engine.run_blast.return_value = mock_hits
        mock_engine_class.return_value = mock_engine
        
        result = search_homologs(
            query_file="query.fasta",
            database="/path/to/db",
            output_file="output.json",
            e_value=1e-5,
            max_hits=100,
            min_identity=50.0,
            min_coverage=70.0
        )
        
        assert result["status"] == "success"
        assert "homolog_set" in result
        
        # Verify engine was called correctly
        mock_engine.run_blast.assert_called_once_with(
            query_file="query.fasta",
            database="/path/to/db",
            e_value=1e-5,
            max_hits=100
        )

    @patch('ampilot.tools.sequence.homology_search.main.BLASTPEngine')
    def test_search_homologs_no_database(self, mock_engine_class):
        """Test homology search with missing database."""
        mock_engine = Mock()
        mock_engine.run_blast.side_effect = FileNotFoundError("Database not found")
        mock_engine_class.return_value = mock_engine
        
        result = search_homologs(
            query_file="query.fasta",
            database="/nonexistent/db",
            output_file="output.json"
        )
        
        assert result["status"] == "error"
        assert "Database not found" in result["error"]

    @patch('ampilot.tools.sequence.homology_search.main.BLASTPEngine')
    def test_search_homologs_blast_error(self, mock_engine_class):
        """Test homology search with BLAST execution error."""
        mock_engine = Mock()
        mock_engine.run_blast.side_effect = RuntimeError("BLAST failed")
        mock_engine_class.return_value = mock_engine
        
        result = search_homologs(
            query_file="query.fasta",
            database="/path/to/db",
            output_file="output.json"
        )
        
        assert result["status"] == "error"
        assert "BLAST failed" in result["error"]

    @patch('ampilot.tools.sequence.homology_search.main.BLASTPEngine')
    def test_search_homologs_filtering(self, mock_engine_class):
        """Test homology search with filtering."""
        # Mock BLAST engine with hits that should be filtered
        mock_engine = Mock()
        mock_hits = [
            HomologyHit(
                query_id="query1", subject_id="subject1", identity=90.0, coverage=95.0,
                e_value=1e-60, bit_score=200.0, alignment_length=100,
                query_start=1, query_end=100, subject_start=1, subject_end=100,
                subject_sequence="GOOD_HIT"
            ),
            HomologyHit(
                query_id="query1", subject_id="subject2", identity=40.0, coverage=50.0,
                e_value=1e-10, bit_score=80.0, alignment_length=50,
                query_start=1, query_end=50, subject_start=1, subject_end=50,
                subject_sequence="BAD_HIT"
            )
        ]
        mock_engine.run_blast.return_value = mock_hits
        mock_engine_class.return_value = mock_engine
        
        result = search_homologs(
            query_file="query.fasta",
            database="/path/to/db",
            output_file="output.json",
            min_identity=70.0,
            min_coverage=80.0
        )
        
        assert result["status"] == "success"
        homolog_set = result["homolog_set"]
        
        # Should only have one hit after filtering
        assert len(homolog_set["results"]) == 1
        assert homolog_set["results"][0]["hits"][0]["identity"] == 90.0
