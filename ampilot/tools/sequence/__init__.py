"""Sequence analysis tools with MCP interface."""

from .ingest_normalize import create_mcp_server as create_ingest_server
from .homology_search import create_mcp_server as create_homology_server  
from .clustering_derep import create_mcp_server as create_clustering_server

__all__ = [
    'create_ingest_server',
    'create_homology_server', 
    'create_clustering_server'
]
