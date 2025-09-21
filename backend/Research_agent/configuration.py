import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure we always load the .env at the repository root
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]  # <repo>/
DOTENV_PATH = REPO_ROOT / ".env"
load_dotenv(DOTENV_PATH)

# OpenAI Configuration
# You also choose another LLM provider，like Google Gemini, Anthropic Claude
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "GRAMPA_Collection")

# Model Configuration
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Data directory at repo root
DATA_DIR = str((REPO_ROOT / "data").resolve())
