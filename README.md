# AMPilot
1. This version of AMPilot belongs to OUC-haide.

2. AMPilot is an intelligent multi-agent system for antimicrobial peptide (AMP) research and analysis. It combines the power of large language models with specialized domain knowledge to provide comprehensive AMP research capabilities.

## 🌟 Features

### 🔬 Research Agent
- **Intelligent Database Search**: Semantic search through comprehensive AMP databases grampa.csv
- **Natural Language Queries**: Ask questions in plain English about antimicrobial peptides
- **Property-based Filtering**: Search by MIC values, target organisms, sequence length, and more
- **Sequence Similarity Search**: Find peptides similar to your query sequence

### 🧬 AMP Rerank (design) Agent  
- **Sequence Analysis**: Comprehensive analysis of peptide sequences and their properties
- **Intelligent Ranking**: AI-powered ranking of peptide sequences based on antimicrobial potential
- **Experience Learning**: Learns from analysis patterns to improve future recommendations
- **Pairwise Comparison**: Detailed comparison between multiple peptide sequences

### 📊 Data Analysis Agent
- **Statistical Analysis**: Perform correlation analysis, regression modeling, and hypothesis testing
- **Natural Language Data Input**: Describe your data in plain English - no file uploads needed
- **Automated Report Generation**: Generate professional PDF(markdown) reports with visualizations
- **Multi-factor Analysis**: Analyze relationships between peptide properties and biological activity

## 🏗️ Architecture

AMPilot uses a sophisticated multi-agent architecture powered by LangGraph:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Research Agent │    │ AMP Designer     │    │ Data Analysis   │
│  (Port 8001)    │    │ Agent            │    │ Agent           │
│                 │    │ (Port 8004)      │    │ (Port 8003)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Manage Agent      │
                    │   (Port 8002)       │
                    │   - Query Routing   │
                    │   - Agent Coordination │
                    │   - Frontend Serving │
                    └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- API Key from LLM provider
### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Xiaoyun-0922/AMPilot.git
   cd AMPilot
   ```

2. **Set up environment**
   ```bash
   conda create -n AMPilot python=3.11
   conda activate AMPilot

   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   # For example, using Openai LLM
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Initialize Research Agent database** (Required for first run)
   ```bash
   cd backend/Research_agent
   python ingest.py
   cd ../..
   ```

### Running AMPilot
```bash
# Start the main management server
cd backend/manage_agent
python api_server.py
```

The system will be available at: **http://127.0.0.1:8002**

## 📖 Usage Examples

### Research Agent Examples

**Find peptides by target organism:**
```
Find antimicrobial peptides effective against Staphylococcus aureus with MIC values less than 10 μM
```

**Sequence property queries:**
```
What are the properties of the peptide KLLKKLLKLWKKLLKKLK?
```

**Database searches:**
```
Show me short antimicrobial peptides (10-15 amino acids) with high activity against gram-positive bacteria
```

### AMP Designer Agent Examples

**Sequence ranking:**
```
Rank these peptide sequences by their antimicrobial potential: KLLKKLLKLWKKLLKKLK, GMWSKILGKLIR, FLPMLAGLAANFLPKIVCKITKKC
```

**Comprehensive analysis:**
```
Analyze and compare the following 10 antimicrobial peptide sequences for activity against S. aureus: [list of sequences]
```

### Data Analysis Agent Examples

**Correlation analysis:**
```
I have antimicrobial peptide data for 10 sequences. Please analyze the correlation between peptide charge and MIC values:

Peptide A: length 12 amino acids, charge +3, MIC against S. aureus is 8.5 μM
Peptide B: length 18 amino acids, charge +6, MIC against S. aureus is 4.2 μM  
[... more data ...]

Please calculate correlation coefficients and provide statistical analysis.
```

**Multi-factor analysis:**
```
Analyze my experimental results for 8 peptides. I want to understand which factors affect their activity:

Peptide 1: 10 amino acids, charge +3, hydrophobicity 0.45, MIC 1.8 μM
[... more data ...]

Perform correlation analysis and generate a comprehensive statistical report.
```

## 🔧 Configuration

### Environment Variables
```bash
# You can also choose another LLM provider, like Google-gemini.
OPENAI_API_KEY=your_openai_api_key

# Optional
MANAGE_AGENT_HOST=127.0.0.1
MANAGE_AGENT_PORT=8002
```

### Agent Ports
- **Manage Agent**: 8002 (main interface)
- **Research Agent**: 8001
- **Data Analysis Agent**: 8003
- **AMP Designer Agent**: 8004

## 📁 Project Structure

```
AMPilot/
├── backend/
│   ├── manage_agent/          # Main coordination agent
│   ├── Research_agent/        # Database search and retrieval
│   ├── AMP_designer_agent/    # Sequence analysis and ranking
│   └── Data_analysis_agent/   # Statistical analysis
├── frontend/                  # Web interface
├── data/                      # Database files
└── README.md
```

## 🛠️ Development

### Adding New Agents
1. Create agent directory in `backend/`
2. Implement FastAPI server with `/health` and `/chat` endpoints
3. Add agent configuration to `backend/manage_agent/configuration.py`
4. Update routing logic in `backend/manage_agent/multi_agent_workflow.py`

### Database Updates
To update the Research Agent database:
```bash
cd backend/Research_agent
python ingest.py
```
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- GRAMPA database for antimicrobial peptide data
- LangGraph for multi-agent orchestration
- FastAPI for robust API framework
- Deepseek & Qwen for large language model inference
- Many thanks to Ziwei Wng for providing guidance during the construction of AMPilot.here
is his github acount: https://github.com/tom-wang813


**AMPilot** - Accelerating antimicrobial peptide research through intelligent AI assistance.
