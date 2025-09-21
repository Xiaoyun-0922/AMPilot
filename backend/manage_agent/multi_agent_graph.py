"""
Thin wrapper exposing MultiAgentGraph for the Manage Agent API server.
This keeps backward-compatibility while the workflow implementation
lives in multi_agent_workflow.py
"""

from typing import Optional, Dict, Any

try:
    from .multi_agent_workflow import MultiAgentWorkflow
except Exception:
    # Fallback when run as a script
    from multi_agent_workflow import MultiAgentWorkflow


class MultiAgentGraph(MultiAgentWorkflow):
    """Graph alias for the multi-agent workflow implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

