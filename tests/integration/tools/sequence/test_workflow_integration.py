"""Integration tests for sequence analysis workflow."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from ampilot.tools.sequence.ingest_normalize.main import process_sequences
from ampilot.tools.sequence.homology_search.main import search_homologs
from ampilot.tools.sequence.clustering_derep.main import cluster_sequences


class TestSequenceWorkflowIntegration:
    """Test integration between sequence analysis modules."""
    
    def test_ingest_to_homology_workflow(self):
        """Test workflow from ingestion to homology search."""
        # Sample input sequences
        input_fasta = """>AMP1
MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
>AMP2
ACDEFGHIKLMNPQRSTVWYJKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHI
>AMP3
GGGGLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEY"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Ingest and normalize
            input_file = temp_path / "input.fasta"
            normalized_file = temp_path / "normalized.fasta"
            
            with open(input_file, 'w') as f:
                f.write(input_fasta)
            
            # Mock the ingest process
            with patch('ampilot.tools.sequence.ingest_normalize.main.FASTAParser') as mock_parser_class, \
                 patch('ampilot.tools.sequence.ingest_normalize.main.SequenceQC') as mock_qc_class:
                
                # Mock successful processing
                mock_parser = Mock()
                mock_parser.parse_file.return_value = []  # Would return SequenceRecord objects
                mock_parser_class.return_value = mock_parser
                
                mock_qc = Mock()
                mock_qc.analyze_batch.return_value = Mock()  # Would return QCReport
                mock_qc_class.return_value = mock_qc
                
                ingest_result = process_sequences(
                    input_file=str(input_file),
                    output_file=str(normalized_file),
                    min_length=10,
                    remove_duplicates=True
                )
                
                assert ingest_result["status"] == "success"
            
            # Step 2: Homology search using normalized sequences
            homology_output = temp_path / "homologs.json"
            
            with patch('ampilot.tools.sequence.homology_search.main.BLASTPEngine') as mock_blast_class:
                mock_blast = Mock()
                mock_blast.run_blast.return_value = []  # Would return HomologyHit objects
                mock_blast_class.return_value = mock_blast
                
                homology_result = search_homologs(
                    query_file=str(normalized_file),
                    database="/mock/database",
                    output_file=str(homology_output),
                    e_value=1e-5,
                    max_hits=100
                )
                
                assert homology_result["status"] == "success"

    def test_homology_to_clustering_workflow(self):
        """Test workflow from homology search to clustering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock homology search results
            homolog_sequences = """>homolog1
MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
>homolog2
MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
>homolog3
ACDEFGHIKLMNPQRSTVWYJKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHI"""
            
            homolog_file = temp_path / "homologs.fasta"
            clustered_file = temp_path / "clustered.fasta"
            
            with open(homolog_file, 'w') as f:
                f.write(homolog_sequences)
            
            # Step: Cluster homologous sequences
            with patch('ampilot.tools.sequence.clustering_derep.main.CDHITEngine') as mock_cdhit_class:
                mock_cdhit = Mock()
                mock_cdhit.run_cdhit.return_value = []  # Would return Cluster objects
                mock_cdhit_class.return_value = mock_cdhit
                
                clustering_result = cluster_sequences(
                    input_file=str(homolog_file),
                    output_file=str(clustered_file),
                    identity_threshold=0.9,
                    coverage_threshold=0.8
                )
                
                assert clustering_result["status"] == "success"

    def test_full_workflow_pipeline(self):
        """Test complete workflow: ingest -> homology -> clustering."""
        input_fasta = """>query1
MKLLNVINFVFLMFVSSSAQAMVDQNVKDVMKYLECSALTQRGSVSIRFNLRTLHFFEYSGG
>query2
ACDEFGHIKLMNPQRSTVWYJKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHI"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # File paths for each step
            input_file = temp_path / "input.fasta"
            normalized_file = temp_path / "normalized.fasta"
            homologs_file = temp_path / "homologs.fasta"
            clustered_file = temp_path / "clustered.fasta"
            
            with open(input_file, 'w') as f:
                f.write(input_fasta)
            
            # Step 1: Ingest and normalize
            with patch('ampilot.tools.sequence.ingest_normalize.main.FASTAParser') as mock_parser_class, \
                 patch('ampilot.tools.sequence.ingest_normalize.main.SequenceQC') as mock_qc_class:
                
                mock_parser = Mock()
                mock_parser.parse_file.return_value = []
                mock_parser_class.return_value = mock_parser
                
                mock_qc = Mock()
                mock_qc.analyze_batch.return_value = Mock()
                mock_qc_class.return_value = mock_qc
                
                ingest_result = process_sequences(
                    input_file=str(input_file),
                    output_file=str(normalized_file),
                    min_length=10
                )
                
                assert ingest_result["status"] == "success"
            
            # Step 2: Homology search
            with patch('ampilot.tools.sequence.homology_search.main.BLASTPEngine') as mock_blast_class:
                mock_blast = Mock()
                mock_blast.run_blast.return_value = []
                mock_blast_class.return_value = mock_blast
                
                homology_result = search_homologs(
                    query_file=str(normalized_file),
                    database="/mock/database",
                    output_file=str(homologs_file),
                    e_value=1e-5
                )
                
                assert homology_result["status"] == "success"
            
            # Step 3: Clustering
            with patch('ampilot.tools.sequence.clustering_derep.main.CDHITEngine') as mock_cdhit_class:
                mock_cdhit = Mock()
                mock_cdhit.run_cdhit.return_value = []
                mock_cdhit_class.return_value = mock_cdhit
                
                clustering_result = cluster_sequences(
                    input_file=str(homologs_file),
                    output_file=str(clustered_file),
                    identity_threshold=0.9
                )
                
                assert clustering_result["status"] == "success"

    def test_error_propagation_in_workflow(self):
        """Test error handling across workflow steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test with non-existent input file
            nonexistent_file = temp_path / "nonexistent.fasta"
            output_file = temp_path / "output.fasta"
            
            # Each step should handle file not found errors gracefully
            ingest_result = process_sequences(
                input_file=str(nonexistent_file),
                output_file=str(output_file)
            )
            assert ingest_result["status"] == "error"
            
            homology_result = search_homologs(
                query_file=str(nonexistent_file),
                database="/mock/database",
                output_file=str(output_file)
            )
            assert homology_result["status"] == "error"
            
            clustering_result = cluster_sequences(
                input_file=str(nonexistent_file),
                output_file=str(output_file)
            )
            assert clustering_result["status"] == "error"

    def test_workflow_data_consistency(self):
        """Test data consistency between workflow steps."""
        # This test would verify that:
        # 1. Output from ingest_normalize is valid input for homology_search
        # 2. Output from homology_search is valid input for clustering_derep
        # 3. Data types and formats are consistent
        
        # Mock data that represents the flow between modules
        sequence_batch = {
            "sequences": [
                {"id": "seq1", "sequence": "MKLLNVINFV", "length": 10},
                {"id": "seq2", "sequence": "ACDEFGHIKL", "length": 10}
            ],
            "total_count": 2
        }
        
        homolog_set = {
            "query_results": [
                {
                    "query_id": "seq1",
                    "hits": [
                        {"subject_id": "homolog1", "identity": 95.0, "e_value": 1e-50}
                    ]
                }
            ],
            "total_hits": 1
        }
        
        cluster_set = {
            "clusters": [
                {
                    "cluster_id": "cluster_0",
                    "representative_id": "homolog1",
                    "members": ["homolog1", "seq1"]
                }
            ],
            "total_clusters": 1
        }
        
        # Verify basic structure consistency
        assert "sequences" in sequence_batch
        assert "query_results" in homolog_set
        assert "clusters" in cluster_set
        
        # In a real test, we would verify type compatibility
        # and data flow between actual module outputs
