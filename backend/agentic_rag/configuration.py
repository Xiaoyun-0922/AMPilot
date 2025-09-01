import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# You can also choose another API provider.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Weaviate Configuration 
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "AMP_Collection")

# Model Configuration 
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")

DATA_DIR = "./data"