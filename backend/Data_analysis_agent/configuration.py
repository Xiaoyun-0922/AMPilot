"""
Configuration file for Data Analysis Agent

LangGraph-based wet lab data analysis agent configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]  # <repo>/
DOTENV_PATH = REPO_ROOT / ".env"
load_dotenv(DOTENV_PATH)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
CHAT_MODEL = "gpt-4o-mini"  # Use gpt-4o-mini model

# Data Analysis Configuration
DATA_DIR = str((CURRENT_FILE.parent / "data").resolve())
RESULTS_DIR = str((CURRENT_FILE.parent / "results").resolve())
TEMP_CODE_DIR = str((CURRENT_FILE.parent / "temp_code").resolve())

# API Configuration
API_HOST = "localhost"
API_PORT = 8003

# Analysis Templates Directory
TEMPLATES_DIR = str((CURRENT_FILE.parent / "templates").resolve())

# Supported file formats
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.txt']

# Maximum file size (MB)
MAX_FILE_SIZE_MB = 100

# Code execution timeout (seconds)
CODE_EXECUTION_TIMEOUT = 300

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, TEMP_CODE_DIR, TEMPLATES_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)
