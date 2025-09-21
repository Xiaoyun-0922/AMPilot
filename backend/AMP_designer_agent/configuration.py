"""
Configuration file for AMP Designer Agent - Version 1

Simplified configuration for:
1. 10 test sequences pairwise comparison
2. Database similarity matching
3. User requirement-based ranking
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

# Model Configuration - Version 1
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Data Configuration
DATA_DIR = str((CURRENT_FILE.parent / "data").resolve())
GRAMPA_DATA_FILE = os.path.join(DATA_DIR, "grampa_with_zscale_properties.csv")
CLUSTERS_DIR = os.path.join(DATA_DIR, "clusters")
EXPERIENCE_DB_PATH = os.path.join(DATA_DIR, "experience_db.json")

# Clustering Configuration - Version 1
MAX_CLUSTER_FILES = None  # No limit on cluster files
SIMILARITY_THRESHOLD = 0.9  # High threshold for quality
MIN_CLUSTER_SIZE = 2

# Experience Learning Configuration
MAX_EXPERIENCES = 1000
EXPERIENCE_SIMILARITY_THRESHOLD = 0.95 

# local api configuration
API_HOST = "127.0.0.1"
API_PORT = 8004

# Ensure necessary directories exist
os.makedirs(CLUSTERS_DIR, exist_ok=True)
