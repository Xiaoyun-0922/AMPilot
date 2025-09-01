"""Unit tests for clustering_derep module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ampilot.tools.sequence.clustering_derep.schemas import (
    ClusterMember, Cluster, ClusterSet, DedupBatch
)
from ampilot.tools.sequence.clustering_derep.cdhit_engine import CDHITEngine
from ampilot.tools.sequence.clustering_derep.main import cluster_sequences


class TestClusterMember:
    """Test ClusterMember data class."""
    
    def test_cluster_member_creation(self):
        """Test creating a cluster member."""
        member = ClusterMember(
            sequence_id="seq1",
            identity_to_representative=95.5,
            coverage=90.0,
            is_representative=True
        )
        
        assert member.sequence_id == "seq1"
        assert member.identity_to_representative == 95.5
        assert member.coverage == 90.0
        assert member.is_representative is True


class TestCluster:
    """Test Cluster data class."""
    
    def test_cluster_creation(self):
        """Test creating a cluster."""
        members = [
            ClusterMember("seq1", 100.0, 100.0, True),
            ClusterMember("seq2", 95.0, 90.0, False),
            ClusterMember("seq3", 92.0, 88.0, False)
        ]
        
        cluster = Cluster(
            cluster_id="cluster_0",
            representative_id="seq1",
            size=3,
            members=members
        )
        
        assert cluster.cluster_id == "cluster_0"
        assert cluster.representative_id == "seq1"
        assert cluster.size == 3
        assert len(cluster.members) == 3
        assert cluster.members[0].is_representative is True


class TestCDHITEngine:
    """Test CD-HIT engine wrapper."""
    
    @patch('subprocess.run')
    def test_run_cdhit_success(self, mock_run):
        """Test successful CD-HIT run."""
        # Mock CD-HIT output files
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "CD-HIT completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock cluster file content
        cluster_content = """>Cluster 0
0	100aa, >seq1... *
1	98aa, >seq2... at 95.5%
2	95aa, >seq3... at 92.0%
>Cluster 1
0	120aa, >seq4... *
1	118aa, >seq5... at 96.0%"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as input_file:
            input_file.write(">seq1\nMKLLNVINFV\n>seq2\nMKLLNVINFX\n>seq3\nMKLLNVINFY")
            input_file.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.clstr', delete=False) as cluster_file:
                cluster_file.write(cluster_content)
                cluster_file.flush()
                
                engine = CDHITEngine()
                
                # Mock the cluster file creation
                with patch.object(engine, '_get_cluster_filename', return_value=cluster_file.name):
                    clusters = engine.run_cdhit(
                        input_file=input_file.name,
                        output_file="output.fasta",
                        identity_threshold=0.9,
                        coverage_threshold=0.8
                    )
                    
                    assert len(clusters) == 2
                    assert clusters[0].cluster_id == "cluster_0"
                    assert clusters[0].size == 3
                    assert clusters[0].representative_id == "seq1"
                    assert clusters[1].cluster_id == "cluster_1"
                    assert clusters[1].size == 2
                    assert clusters[1].representative_id == "seq4"
                
        Path(input_file.name).unlink()
        Path(cluster_file.name).unlink()

    @patch('subprocess.run')
    def test_run_cdhit_failure(self, mock_run):
        """Test CD-HIT run failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Input file not found"
        mock_run.return_value = mock_result
        
        engine = CDHITEngine()
        with pytest.raises(RuntimeError, match="CD-HIT failed"):
            engine.run_cdhit(
                input_file="nonexistent.fasta",
                output_file="output.fasta",
                identity_threshold=0.9
            )

    def test_parse_cluster_file(self):
        """Test parsing CD-HIT cluster file."""
        cluster_content = """>Cluster 0
0	100aa, >seq1... *
1	98aa, >seq2... at 95.5%
2	95aa, >seq3... at 92.0%
>Cluster 1
0	120aa, >seq4... *
1	118aa, >seq5... at 96.0%"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.clstr', delete=False) as cluster_file:
            cluster_file.write(cluster_content)
            cluster_file.flush()
            
            engine = CDHITEngine()
            clusters = engine._parse_cluster_file(cluster_file.name)
            
            assert len(clusters) == 2
            
            # Check first cluster
            cluster1 = clusters[0]
            assert cluster1.cluster_id == "cluster_0"
            assert cluster1.representative_id == "seq1"
            assert cluster1.size == 3
            assert len(cluster1.members) == 3
            
            # Check representative member
            rep_member = next(m for m in cluster1.members if m.is_representative)
            assert rep_member.sequence_id == "seq1"
            assert rep_member.identity_to_representative == 100.0
            
            # Check non-representative members
            non_rep_members = [m for m in cluster1.members if not m.is_representative]
            assert len(non_rep_members) == 2
            assert non_rep_members[0].sequence_id == "seq2"
            assert non_rep_members[0].identity_to_representative == 95.5
            
            # Check second cluster
            cluster2 = clusters[1]
            assert cluster2.cluster_id == "cluster_1"
            assert cluster2.representative_id == "seq4"
            assert cluster2.size == 2
            
        Path(cluster_file.name).unlink()

    def test_parse_malformed_cluster_file(self):
        """Test parsing malformed cluster file."""
        malformed_content = """>Cluster 0
0	100aa, >seq1... *
Invalid line format
1	98aa, >seq2... at 95.5%"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.clstr', delete=False) as cluster_file:
            cluster_file.write(malformed_content)
            cluster_file.flush()
            
            engine = CDHITEngine()
            with pytest.raises(ValueError, match="Invalid cluster file format"):
                engine._parse_cluster_file(cluster_file.name)
                
        Path(cluster_file.name).unlink()

    def test_extract_sequence_info(self):
        """Test extracting sequence info from cluster line."""
        engine = CDHITEngine()
        
        # Test representative sequence
        line = "0\t100aa, >seq1... *"
        seq_id, identity, is_rep = engine._extract_sequence_info(line)
        assert seq_id == "seq1"
        assert identity == 100.0
        assert is_rep is True
        
        # Test non-representative sequence
        line = "1\t98aa, >seq2... at 95.5%"
        seq_id, identity, is_rep = engine._extract_sequence_info(line)
        assert seq_id == "seq2"
        assert identity == 95.5
        assert is_rep is False
        
        # Test with complex sequence ID
        line = "2\t95aa, >gi|123456|ref|NP_000001.1|... at 92.0%"
        seq_id, identity, is_rep = engine._extract_sequence_info(line)
        assert seq_id == "gi|123456|ref|NP_000001.1|"
        assert identity == 92.0
        assert is_rep is False


class TestClusterSequences:
    """Test main clustering function."""
    
    @patch('ampilot.tools.sequence.clustering_derep.main.CDHITEngine')
    def test_cluster_sequences_success(self, mock_engine_class):
        """Test successful sequence clustering."""
        # Mock CD-HIT engine
        mock_engine = Mock()
        mock_clusters = [
            Cluster(
                cluster_id="cluster_0",
                representative_id="seq1",
                size=2,
                members=[
                    ClusterMember("seq1", 100.0, 100.0, True),
                    ClusterMember("seq2", 95.0, 90.0, False)
                ]
            )
        ]
        mock_engine.run_cdhit.return_value = mock_clusters
        mock_engine_class.return_value = mock_engine
        
        result = cluster_sequences(
            input_file="input.fasta",
            output_file="output.fasta",
            identity_threshold=0.9,
            coverage_threshold=0.8,
            word_length=5
        )
        
        assert result["status"] == "success"
        assert "cluster_set" in result
        assert "dedup_batch" in result
        
        # Verify engine was called correctly
        mock_engine.run_cdhit.assert_called_once_with(
            input_file="input.fasta",
            output_file="output.fasta",
            identity_threshold=0.9,
            coverage_threshold=0.8,
            word_length=5,
            memory_limit=2000
        )

    @patch('ampilot.tools.sequence.clustering_derep.main.CDHITEngine')
    def test_cluster_sequences_file_error(self, mock_engine_class):
        """Test clustering with file error."""
        mock_engine = Mock()
        mock_engine.run_cdhit.side_effect = FileNotFoundError("Input file not found")
        mock_engine_class.return_value = mock_engine
        
        result = cluster_sequences(
            input_file="nonexistent.fasta",
            output_file="output.fasta"
        )
        
        assert result["status"] == "error"
        assert "Input file not found" in result["error"]

    @patch('ampilot.tools.sequence.clustering_derep.main.CDHITEngine')
    def test_cluster_sequences_cdhit_error(self, mock_engine_class):
        """Test clustering with CD-HIT execution error."""
        mock_engine = Mock()
        mock_engine.run_cdhit.side_effect = RuntimeError("CD-HIT failed")
        mock_engine_class.return_value = mock_engine
        
        result = cluster_sequences(
            input_file="input.fasta",
            output_file="output.fasta"
        )
        
        assert result["status"] == "error"
        assert "CD-HIT failed" in result["error"]

    @patch('ampilot.tools.sequence.clustering_derep.main.CDHITEngine')
    def test_cluster_sequences_empty_result(self, mock_engine_class):
        """Test clustering with empty result."""
        mock_engine = Mock()
        mock_engine.run_cdhit.return_value = []  # No clusters found
        mock_engine_class.return_value = mock_engine
        
        result = cluster_sequences(
            input_file="input.fasta",
            output_file="output.fasta"
        )
        
        assert result["status"] == "success"
        cluster_set = result["cluster_set"]
        assert cluster_set["total_clusters"] == 0
        assert len(cluster_set["clusters"]) == 0

    @patch('ampilot.tools.sequence.clustering_derep.main.CDHITEngine')
    def test_cluster_sequences_statistics(self, mock_engine_class):
        """Test clustering statistics calculation."""
        mock_engine = Mock()
        mock_clusters = [
            Cluster("cluster_0", "seq1", 3, [
                ClusterMember("seq1", 100.0, 100.0, True),
                ClusterMember("seq2", 95.0, 90.0, False),
                ClusterMember("seq3", 92.0, 88.0, False)
            ]),
            Cluster("cluster_1", "seq4", 2, [
                ClusterMember("seq4", 100.0, 100.0, True),
                ClusterMember("seq5", 96.0, 85.0, False)
            ])
        ]
        mock_engine.run_cdhit.return_value = mock_clusters
        mock_engine_class.return_value = mock_engine
        
        result = cluster_sequences(
            input_file="input.fasta",
            output_file="output.fasta"
        )
        
        assert result["status"] == "success"
        cluster_set = result["cluster_set"]
        
        assert cluster_set["total_clusters"] == 2
        assert cluster_set["total_sequences"] == 5
        assert cluster_set["representatives_count"] == 2
        assert cluster_set["largest_cluster_size"] == 3
        assert cluster_set["mean_cluster_size"] == 2.5
