"""
Configuration file for Multi-Agent Management System

This module contains all configuration settings for the AMPilot multi-agent system,
including API endpoints, model settings, and agent configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure we always load the .env at the repository root
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]  # <repo>/
DOTENV_PATH = REPO_ROOT / ".env"
load_dotenv(DOTENV_PATH)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
CHAT_MODEL = "gpt-4o-mini"
ROUTING_MODEL = "gpt-4o-mini"  # Lightweight model for routing decisions
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Router behavior flags
# If True, do not use any local keyword fallback; require LLM routing
ROUTER_REQUIRE_LLM = True
# If True, disable keyword-based intelligent fallback routing
ROUTER_DISABLE_KEYWORDS = True

# Main API Server Configuration
API_HOST = "0.0.0.0"
API_PORT = 8001
CORS_ORIGINS = ["*"]  # In production, specify exact origins

# Agent Service Endpoints
RESEARCH_AGENT_URL = "http://localhost:8001"
AMP_DESIGNER_AGENT_URL = "http://localhost:8004"
DATA_ANALYSIS_AGENT_URL = "http://localhost:8003"  # Align with Data Analysis agent config

# Agent Service Ports (for reference)
RESEARCH_AGENT_PORT = 8001
AMP_DESIGNER_AGENT_PORT = 8004
DATA_ANALYSIS_AGENT_PORT = 8003

# File Upload Configuration
UPLOAD_DIR = str((CURRENT_FILE.parent / "uploads").resolve())
MAX_FILE_SIZE_MB = 100
SUPPORTED_FILE_FORMATS = [".csv", ".xlsx", ".xls", ".json", ".txt"]

# Frontend Configuration
FRONTEND_DIR = str((REPO_ROOT / "frontend").resolve())

# Agent Routing Configuration
MAX_ROUTING_ATTEMPTS = 3

# WebSocket Configuration
WEBSOCKET_TIMEOUT = 300  # 5 minutes
WEBSOCKET_PING_INTERVAL = 30  # 30 seconds

# Session Management
SESSION_TIMEOUT_MINUTES = 60
MAX_SESSIONS = 1000

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Agent Capabilities Mapping
AGENT_CAPABILITIES = {
    "research": {
        "name": "Research Agent",
        "description": "Retrieves AMP information from database using semantic search",
        "capabilities": [
            "sequence_to_properties",
            "properties_to_sequence", 
            "general_amp_knowledge",
            "database_search"
        ],
        "url": RESEARCH_AGENT_URL,
        "health_endpoint": "/health"
    },
    "amp_designer": {
        "name": "AMP Designer Agent", 
        "description": "Analyzes and ranks antimicrobial peptide sequences",
        "capabilities": [
            "sequence_analysis",
            "sequence_ranking",
            "pairwise_comparison",
            "experience_learning"
        ],
        "url": AMP_DESIGNER_AGENT_URL,
        "health_endpoint": "/health"
    },
    "data_analysis": {
        "name": "Data Analysis Agent",
        "description": "Performs statistical analysis on experimental data",
        "capabilities": [
            "statistical_analysis",
            "data_visualization", 
            "hypothesis_testing",
            "correlation_analysis",
            "code_generation"
        ],
        "url": DATA_ANALYSIS_AGENT_URL,
        "health_endpoint": "/health"
    }
}



# Default Responses
DEFAULT_RESPONSES = {
    "welcome": """Hello! 👋

I'm your specialized AI assistant for antimicrobial peptide wet lab. You can ask me to:

🔬 **Find AMPs** - Search peptides by target bacteria, MIC values, or properties
🧬 **Rank sequences** - Compare and rank your peptide sequences by antimicrobial potential
📊 **Analyze data** - Interpret your experimental results and activity data

What would you like to explore today?""",
    
    "routing_failed": """
    I couldn't complete this request because the specialized agent wasn't reachable just now.

    Try again in a moment, or verify the services are running:
    - Research agent (http://127.0.0.1:8001)
    - AMP designer agent (http://127.0.0.1:8004)
    - Data analysis agent (http://127.0.0.1:8003)

    Example queries:
    - Find peptides active against S. aureus with MIC < 10 µM
    - What are the properties of the peptide: LPLLAGLAANFLPKIFCKITRK
    - Analyze correlation between peptide charge and MIC values
    """,

    "agent_unavailable": """
    I apologize, but the requested service is currently unavailable. 
    Please try again later or contact support if the issue persists.
    """
}

# Ensure necessary directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
