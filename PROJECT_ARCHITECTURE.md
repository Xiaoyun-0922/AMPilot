# AMPilot - Antimicrobial Peptide Research Assistant

## Project Overview

AMPilot is an intelligent research assistant for Antimicrobial Peptides (AMPs) that combines Retrieval-Augmented Generation (RAG) with LangGraph to provide comprehensive information about AMPs from a structured database.

## Architecture Diagram

```
AMPilot/
├── backend/
│   ├── agentic_rag/           # Main RAG system
│   │   ├── __init__.py
│   │   ├── configuration.py   # Environment and API configuration
│   │   ├── main.py           # Entry point for chat interface
│   │   ├── embeddings.py     # BGE embedding model setup
│   │   ├── ingest.py         # Data ingestion to Weaviate
│   │   ├── retrieval.py      # Vector search functionality
│   │   ├── debug.py          # Debug utilities
│   │   └── retrieval_graph/  # LangGraph workflow
│   │       ├── graph.py      # Main graph definition
│   │       ├── state.py      # Agent state management
│   │       └── prompts.py    # System prompts
│   └── amp_predicter/        # Future AMP prediction module (empty)
├── data/
│   └── AMPdata.csv          # AMP database (36,000+ records)
├── pyproject.toml           # Project dependencies
└── requirments.txt          # Requirements file
```

## System Components

### 1. Data Layer
- **AMPdata.csv**: Contains 36,000+ AMP records with fields:
  - `DRAMP_ID`: Unique identifier
  - `Sequence`: Peptide sequence
  - `Name`: Peptide name
  - `Description`: Functional description
  - `Reference`: Scientific references

### 2. Vector Database (Weaviate)
- **Purpose**: Stores embedded AMP data for semantic search
- **Collection**: `AMP_Collection`
- **Embeddings**: BGE-large-en-v1.5 model
- **Configuration**: Local instance (localhost:8080)

### 3. RAG System (agentic_rag/)

#### Core Modules:

**configuration.py**
- Environment variable management
- API keys (OpenRouter, OpenAI)
- Weaviate connection settings
- Model configuration

**embeddings.py**
- BGE embedding model initialization
- CUDA-optimized for performance
- Normalized embeddings for better similarity search

**ingest.py**
- CSV data processing and cleaning
- Text preparation for embedding
- Batch import to Weaviate
- Progress tracking with tqdm

**retrieval.py**
- Vector similarity search
- Query embedding generation
- Result formatting and error handling

#### LangGraph Workflow (retrieval_graph/):

**graph.py**
- StateGraph definition
- Agent and tool nodes
- Conditional routing logic
- Tool binding and execution

**state.py**
- AgentState TypedDict
- Message history management

**prompts.py**
- System prompt for AMP expert persona
- JSON output format specification
- Structured response guidelines

### 4. User Interface

**main.py**
- Interactive command-line chat
- Real-time streaming responses
- Session management
- Error handling

## Data Flow

1. **Data Ingestion**:
   ```
   CSV Files → Text Processing → Embedding Generation → Weaviate Storage
   ```

2. **Query Processing**:
   ```
   User Query → Embedding → Vector Search → Context Retrieval → LLM Response
   ```

3. **LangGraph Workflow**:
   ```
   Agent Node → Tool Decision → Retrieval Tool → Response Generation → End
   ```

## Technology Stack

- **LLM**: GPT-4o-mini (via OpenRouter)
- **Embeddings**: BGE-large-en-v1.5 (HuggingFace)
- **Vector DB**: Weaviate
- **Framework**: LangGraph + LangChain
- **Language**: Python 3.x

## Key Features

1. **Semantic Search**: Advanced vector similarity search for AMP information
2. **Structured Responses**: JSON-formatted outputs with confidence levels
3. **Real-time Streaming**: Token-by-token response generation
4. **Expert Persona**: Specialized AMP research assistant
5. **Scalable Architecture**: Modular design for easy extension

## Future Extensions

- **amp_predicter/**: Planned module for AMP property prediction
- **Web Interface**: GUI for broader accessibility
- **API Endpoints**: RESTful API for integration
- **Advanced Analytics**: Statistical analysis and visualization

## Setup Requirements

1. Python environment with required dependencies
2. Weaviate instance running locally
3. API keys for OpenRouter/OpenAI
4. CUDA-enabled GPU (recommended for embeddings)
5. AMP data in CSV format

## Usage

```bash
# Start the chat interface
python backend/agentic_rag/main.py

# Ingest data to Weaviate
python backend/agentic_rag/ingest.py
```

This architecture provides a robust foundation for AMP research assistance with room for future enhancements and scaling.
