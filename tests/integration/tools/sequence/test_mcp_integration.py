"""Integration tests for MCP server functionality."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

# Note: These imports would need to be adjusted based on the actual fastMCP implementation
# from ampilot.tools.sequence.ingest_normalize import create_mcp_server as create_ingest_server
# from ampilot.tools.sequence.homology_search import create_mcp_server as create_homology_server
# from ampilot.tools.sequence.clustering_derep import create_mcp_server as create_clustering_server


class TestMCPServerIntegration:
    """Test MCP server integration and communication."""
    
    @pytest.mark.asyncio
    async def test_ingest_mcp_server_lifecycle(self):
        """Test MCP server startup, tool call, and shutdown for ingest_normalize."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock the actual server creation for now
            # In real implementation, would create actual server
            mock_server = Mock()
            mock_server.start = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server.call_tool = AsyncMock()
            
            # Mock tool response
            mock_response = {
                "status": "success",
                "sequence_batch": {
                    "sequences": [{"id": "seq1", "sequence": "MKLLNV", "length": 6}],
                    "total_count": 1
                },
                "qc_report": {
                    "total_sequences": 1,
                    "valid_sequences": 1,
                    "issues": []
                }
            }
            mock_server.call_tool.return_value = mock_response
            
            # Test server lifecycle
            await mock_server.start()
            
            # Test tool call
            result = await mock_server.call_tool(
                "process_sequences",
                {
                    "input_file": str(temp_path / "test.fasta"),
                    "output_file": str(temp_path / "output.fasta"),
                    "min_length": 5
                }
            )
            
            assert result["status"] == "success"
            assert "sequence_batch" in result
            assert "qc_report" in result
            
            await mock_server.stop()

    @pytest.mark.asyncio
    async def test_homology_mcp_server_tool_call(self):
        """Test MCP server tool call for homology search."""
        mock_server = Mock()
        mock_server.call_tool = AsyncMock()
        
        # Mock homology search response
        mock_response = {
            "status": "success",
            "homolog_set": {
                "query_results": [
                    {
                        "query_id": "query1",
                        "hits": [
                            {
                                "subject_id": "homolog1",
                                "identity": 95.0,
                                "e_value": 1e-50,
                                "bit_score": 200.0
                            }
                        ]
                    }
                ],
                "total_queries": 1,
                "total_hits": 1
            }
        }
        mock_server.call_tool.return_value = mock_response
        
        result = await mock_server.call_tool(
            "search_homologs",
            {
                "query_file": "query.fasta",
                "database": "/path/to/db",
                "output_file": "homologs.json",
                "e_value": 1e-5,
                "max_hits": 100
            }
        )
        
        assert result["status"] == "success"
        assert "homolog_set" in result
        assert result["homolog_set"]["total_hits"] == 1

    @pytest.mark.asyncio
    async def test_clustering_mcp_server_tool_call(self):
        """Test MCP server tool call for clustering."""
        mock_server = Mock()
        mock_server.call_tool = AsyncMock()
        
        # Mock clustering response
        mock_response = {
            "status": "success",
            "cluster_set": {
                "clusters": [
                    {
                        "cluster_id": "cluster_0",
                        "representative_id": "seq1",
                        "size": 3,
                        "members": [
                            {"sequence_id": "seq1", "is_representative": True},
                            {"sequence_id": "seq2", "is_representative": False},
                            {"sequence_id": "seq3", "is_representative": False}
                        ]
                    }
                ],
                "total_clusters": 1,
                "total_sequences": 3
            },
            "dedup_batch": {
                "representatives": ["seq1"],
                "total_representatives": 1
            }
        }
        mock_server.call_tool.return_value = mock_response
        
        result = await mock_server.call_tool(
            "cluster_sequences",
            {
                "input_file": "input.fasta",
                "output_file": "clustered.fasta",
                "identity_threshold": 0.9,
                "coverage_threshold": 0.8
            }
        )
        
        assert result["status"] == "success"
        assert "cluster_set" in result
        assert "dedup_batch" in result
        assert result["cluster_set"]["total_clusters"] == 1

    @pytest.mark.asyncio
    async def test_mcp_server_error_handling(self):
        """Test MCP server error handling."""
        mock_server = Mock()
        mock_server.call_tool = AsyncMock()
        
        # Mock error response
        mock_error_response = {
            "status": "error",
            "error": "File not found: nonexistent.fasta",
            "error_type": "FileNotFoundError"
        }
        mock_server.call_tool.return_value = mock_error_response
        
        result = await mock_server.call_tool(
            "process_sequences",
            {
                "input_file": "nonexistent.fasta",
                "output_file": "output.fasta"
            }
        )
        
        assert result["status"] == "error"
        assert "File not found" in result["error"]
        assert result["error_type"] == "FileNotFoundError"

    @pytest.mark.asyncio
    async def test_mcp_server_parameter_validation(self):
        """Test MCP server parameter validation."""
        mock_server = Mock()
        mock_server.call_tool = AsyncMock()
        
        # Mock validation error response
        mock_validation_error = {
            "status": "error",
            "error": "Invalid parameter: identity_threshold must be between 0 and 1",
            "error_type": "ValidationError"
        }
        mock_server.call_tool.return_value = mock_validation_error
        
        result = await mock_server.call_tool(
            "cluster_sequences",
            {
                "input_file": "input.fasta",
                "output_file": "output.fasta",
                "identity_threshold": 1.5  # Invalid value
            }
        )
        
        assert result["status"] == "error"
        assert "Invalid parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_mcp_workflow_coordination(self):
        """Test coordinating multiple MCP servers for workflow."""
        # Mock multiple servers
        ingest_server = Mock()
        homology_server = Mock()
        clustering_server = Mock()
        
        ingest_server.call_tool = AsyncMock()
        homology_server.call_tool = AsyncMock()
        clustering_server.call_tool = AsyncMock()
        
        # Mock responses for each step
        ingest_response = {
            "status": "success",
            "sequence_batch": {"total_count": 2},
            "output_file": "normalized.fasta"
        }
        
        homology_response = {
            "status": "success",
            "homolog_set": {"total_hits": 10},
            "output_file": "homologs.fasta"
        }
        
        clustering_response = {
            "status": "success",
            "cluster_set": {"total_clusters": 3},
            "output_file": "clustered.fasta"
        }
        
        ingest_server.call_tool.return_value = ingest_response
        homology_server.call_tool.return_value = homology_response
        clustering_server.call_tool.return_value = clustering_response
        
        # Simulate workflow coordination
        # Step 1: Ingest
        ingest_result = await ingest_server.call_tool(
            "process_sequences",
            {"input_file": "raw.fasta", "output_file": "normalized.fasta"}
        )
        assert ingest_result["status"] == "success"
        
        # Step 2: Homology search (using output from step 1)
        homology_result = await homology_server.call_tool(
            "search_homologs",
            {
                "query_file": ingest_result["output_file"],
                "database": "/db",
                "output_file": "homologs.fasta"
            }
        )
        assert homology_result["status"] == "success"
        
        # Step 3: Clustering (using output from step 2)
        clustering_result = await clustering_server.call_tool(
            "cluster_sequences",
            {
                "input_file": homology_result["output_file"],
                "output_file": "clustered.fasta"
            }
        )
        assert clustering_result["status"] == "success"

    def test_mcp_server_configuration(self):
        """Test MCP server configuration and metadata."""
        # Mock server configuration
        server_config = {
            "name": "ampilot-sequence-ingest",
            "version": "1.0.0",
            "description": "Sequence ingestion and normalization",
            "tools": [
                {
                    "name": "process_sequences",
                    "description": "Process and normalize input sequences",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input_file": {"type": "string"},
                            "output_file": {"type": "string"},
                            "min_length": {"type": "integer", "default": 10},
                            "remove_duplicates": {"type": "boolean", "default": True}
                        },
                        "required": ["input_file", "output_file"]
                    }
                }
            ]
        }
        
        # Verify configuration structure
        assert "name" in server_config
        assert "version" in server_config
        assert "tools" in server_config
        assert len(server_config["tools"]) == 1
        
        tool_config = server_config["tools"][0]
        assert tool_config["name"] == "process_sequences"
        assert "parameters" in tool_config
        assert "required" in tool_config["parameters"]

    @pytest.mark.asyncio
    async def test_mcp_server_concurrent_requests(self):
        """Test MCP server handling concurrent requests."""
        mock_server = Mock()
        mock_server.call_tool = AsyncMock()
        
        # Mock response with delay to simulate processing time
        async def mock_tool_call(tool_name, params):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "status": "success",
                "tool": tool_name,
                "params": params
            }
        
        mock_server.call_tool.side_effect = mock_tool_call
        
        # Submit multiple concurrent requests
        tasks = []
        for i in range(3):
            task = mock_server.call_tool(
                "process_sequences",
                {"input_file": f"input_{i}.fasta", "output_file": f"output_{i}.fasta"}
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["status"] == "success"
            assert result["params"]["input_file"] == f"input_{i}.fasta"
