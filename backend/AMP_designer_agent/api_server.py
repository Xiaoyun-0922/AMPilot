"""
AMP Designer Agent API Server - Version 1
FastAPI-based REST API for antimicrobial peptide analysis and ranking

Features:
1. 10 test sequences pairwise comparison
2. Database similarity matching
3. User requirement-based ranking (bacteria type, MIC range, properties)
4. Experience learning and management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure local package imports work when run via different entry points
import sys
from pathlib import Path
_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

try:
    from .configuration import API_HOST, API_PORT, CLUSTERS_DIR, GRAMPA_DATA_FILE
    from .tools.sequence_clustering import SequenceDataProcessor
    from .tools.sequence_embedder import SequenceEmbedder
    from .graph.workflow import AMPAnalysisWorkflow
    from .graph.experience_manager import ExperienceManager
    from .graph.enhanced_training_module import EnhancedAMPTrainingModule as AMPTrainingModule, EnhancedTrainingConfig as TrainingConfig
except Exception:
    from configuration import API_HOST, API_PORT, CLUSTERS_DIR, GRAMPA_DATA_FILE
    from tools.sequence_clustering import SequenceDataProcessor
    from tools.sequence_embedder import SequenceEmbedder
    from graph.workflow import AMPAnalysisWorkflow
    from graph.experience_manager import ExperienceManager
    from graph.enhanced_training_module import EnhancedAMPTrainingModule as AMPTrainingModule, EnhancedTrainingConfig as TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AMP Designer Agent API - Version 1",
    description="Antimicrobial peptide sequence analysis and ranking system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
workflow: Optional[AMPAnalysisWorkflow] = None
data_processor: Optional[SequenceDataProcessor] = None
experience_manager: Optional[ExperienceManager] = None
training_module: Optional[AMPTrainingModule] = None
test_sequences: List[str] = []
sequence_database: Optional[pd.DataFrame] = None

# Test sequences for pairwise comparison
DEFAULT_TEST_SEQUENCES = [
    "KLLKKLLKLWKKLLKKLK",           # Cationic, high lysine content
    "FLPMLAGLAANFLPKIVCKITKKC",     # Mixed hydrophobic/cationic
    "GMWSKILGKLIR",                 # Short, balanced
    "KKLLKWKLKLKK",                 # Cationic, tryptophan
    "GNNRPVYIAQPRPPHPAL",           # Proline-rich
    "FLPFIAGMAAKFLPKIFCAISKKC",     # Hydrophobic-rich
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Long, complex
    "GIGKFLHSAKKFGKAFVGEIMNS",      # Balanced composition
    "KLKLLLLLKLK",                  # Simple repeat pattern
    "RWRWRWRWRW"                    # Alternating arginine-tryptophan
]

# Pydantic models
class UserRequirements(BaseModel):
    target_bacteria: Optional[List[str]] = Field(None, description="Target bacterial strains")
    mic_range: Optional[Tuple[float, float]] = Field(None, description="MIC range (min, max) in μg/mL")
    min_activity: Optional[float] = Field(None, description="Minimum antimicrobial activity")
    max_toxicity: Optional[float] = Field(None, description="Maximum toxicity level")
    physicochemical_properties: Optional[Dict[str, Any]] = Field(None, description="Desired physicochemical properties")
    sequence_length_range: Optional[Tuple[int, int]] = Field(None, description="Sequence length range")

class SequenceRankingRequest(BaseModel):
    sequences: Optional[List[str]] = Field(None, description="Sequences to rank (default: use test sequences)")
    requirements: UserRequirements = Field(default_factory=UserRequirements)
    include_database_matches: bool = Field(True, description="Include similar sequences from database")
    similarity_threshold: float = Field(0.8, description="Similarity threshold for database matching")

class ChatMessage(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None

class PairwiseAnalysisRequest(BaseModel):
    sequences: Optional[List[str]] = Field(None, description="Sequences for pairwise analysis")
    max_pairs: int = Field(45, description="Maximum number of pairs to analyze (default: all pairs from 10 sequences)")

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    insights: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    max_pairs: Optional[int] = Field(50, description="Maximum number of sequence pairs to analyze")
    similarity_threshold: Optional[float] = Field(0.85, description="Similarity threshold for sequence pairs")
    batch_size: Optional[int] = Field(10, description="Batch size for processing")

class TrainingResponse(BaseModel):
    success: bool
    message: str
    session_stats: Optional[Dict[str, Any]] = None
    overall_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global workflow, data_processor, experience_manager, training_module, test_sequences, sequence_database

    try:
        logger.info("Initializing AMP Designer Agent API Server - Version 1...")

        # Initialize components
        workflow = AMPAnalysisWorkflow()
        data_processor = SequenceDataProcessor()
        experience_manager = ExperienceManager()
        training_module = AMPTrainingModule()

        # Load test sequences
        test_sequences = DEFAULT_TEST_SEQUENCES.copy()

        # Load sequence database
        if os.path.exists(GRAMPA_DATA_FILE):
            sequence_database = pd.read_csv(GRAMPA_DATA_FILE)
            logger.info(f"Loaded {len(sequence_database)} sequences from database")
        else:
            logger.warning(f"Database file not found: {GRAMPA_DATA_FILE}")
            sequence_database = pd.DataFrame()

        logger.info("AMP Designer Agent API Server initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API server: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AMP Designer Agent API Server - Version 1",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "10 test sequences pairwise comparison",
            "Database similarity matching",
            "User requirement-based ranking",
            "Experience learning and management"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "workflow_initialized": workflow is not None,
        "training_module_initialized": training_module is not None,
        "sequences_loaded": len(test_sequences)
    }

@app.get("/test_sequences")
async def get_test_sequences():
    """Get the current test sequences"""
    return {
        "sequences": test_sequences,
        "count": len(test_sequences),
        "description": "Default test sequences for pairwise comparison"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Main chat endpoint for AMP Designer Agent.
    Routes natural language queries to appropriate analysis functions.
    """
    try:
        message = chat_message.message.lower()

        # Determine what type of analysis to perform based on the message
        if any(keyword in message for keyword in ['rank', 'ranking', 'compare', 'best', 'activity']):
            # Extract sequences from the message if provided
            sequences_to_analyze = extract_sequences_from_message(chat_message.message)

            if sequences_to_analyze:
                # Use sequences from the message
                target_sequences = sequences_to_analyze
            else:
                # Use default test sequences
                target_sequences = test_sequences

            # Perform comprehensive ranking analysis with LLM guidance
            detailed_analysis = await perform_comprehensive_ranking_analysis(
                sequences=target_sequences,
                user_query=chat_message.message,
                session_id=chat_message.session_id
            )

            return ChatResponse(
                response=detailed_analysis["response"],
                metadata={
                    "agent": "amp_designer",
                    "analysis_type": "comprehensive_ranking",
                    "success": detailed_analysis["success"],
                    "session_id": chat_message.session_id,
                    "sequences_analyzed": len(target_sequences),
                    "analysis_steps": detailed_analysis.get("analysis_steps", [])
                }
            )

        elif any(keyword in message for keyword in ['analyze', 'analysis', 'pairwise', 'similarity']):
            # Use pairwise analysis endpoint
            request = PairwiseAnalysisRequest(
                sequences=None,  # Use default test sequences
                max_pairs=10
            )
            result = await analyze_pairwise(request)

            return ChatResponse(
                response=result.message,
                metadata={
                    "agent": "amp_designer",
                    "analysis_type": "pairwise",
                    "success": result.success,
                    "session_id": chat_message.session_id
                }
            )

        else:
            # Default response with capabilities
            return ChatResponse(
                response="""I'm the AMP Designer Agent. I can help you analyze and rank antimicrobial peptide sequences.

**My capabilities:**
- **Sequence ranking** - Rank sequences by predicted antimicrobial activity
- **Pairwise analysis** - Compare sequences for similarity and properties
- **Database matching** - Find similar sequences in our database
- **Activity prediction** - Predict antimicrobial activity based on sequence properties

**Example queries:**
- "Rank these sequences by activity"
- "Analyze sequence similarities"
- "Compare sequences for antimicrobial potential"

How can I help you with sequence analysis?""",
                metadata={
                    "agent": "amp_designer",
                    "analysis_type": "info",
                    "session_id": chat_message.session_id
                }
            )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error processing your request: {str(e)}",
            metadata={
                "agent": "amp_designer",
                "error": str(e),
                "session_id": chat_message.session_id
            }
        )

@app.post("/test_sequences")
async def update_test_sequences(sequences: List[str]):
    """Update the test sequences"""
    global test_sequences

    if len(sequences) < 2:
        raise HTTPException(status_code=400, detail="At least 2 sequences required")

    test_sequences = sequences
    return {
        "message": f"Updated test sequences to {len(sequences)} sequences",
        "sequences": test_sequences
    }

@app.post("/analysis/pairwise", response_model=AnalysisResponse)
async def analyze_pairwise(request: PairwiseAnalysisRequest):
    """Perform pairwise analysis of sequences"""
    try:
        sequences = request.sequences or test_sequences

        if len(sequences) < 2:
            raise HTTPException(status_code=400, detail="At least 2 sequences required for pairwise analysis")

        # Generate all possible pairs
        pairs = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                pairs.append((sequences[i], sequences[j]))

        # Limit pairs if requested
        if request.max_pairs < len(pairs):
            pairs = pairs[:request.max_pairs]

        results = []
        insights_collected = []

        for i, (seq1, seq2) in enumerate(pairs):
            logger.info(f"Analyzing pair {i+1}/{len(pairs)}: {seq1[:10]}... vs {seq2[:10]}...")

            # Analyze using workflow
            result = workflow.analyze_sequence_pair(seq1, seq2)

            if result.get('success'):
                pair_result = {
                    "pair_id": i + 1,
                    "sequence1": seq1,
                    "sequence2": seq2,
                    "insights": result.get('insights', []),
                    "analysis_details": result.get('analysis_details', {})
                }
                results.append(pair_result)
                insights_collected.extend(result.get('insights', []))
            else:
                logger.warning(f"Failed to analyze pair {i+1}: {result.get('error')}")

        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed {len(results)} sequence pairs",
            data={
                "pairs_analyzed": len(results),
                "total_pairs_possible": len(pairs),
                "results": results
            },
            insights=insights_collected
        )

    except Exception as e:
        logger.error(f"Pairwise analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ranking/sequences", response_model=AnalysisResponse)
async def rank_sequences(request: SequenceRankingRequest):
    """Rank sequences based on user requirements"""
    try:
        sequences = request.sequences or test_sequences

        if len(sequences) < 2:
            raise HTTPException(status_code=400, detail="At least 2 sequences required for ranking")

        # Find similar sequences from database if requested
        if request.include_database_matches and not sequence_database.empty:
            similar_sequences = await find_similar_sequences(
                sequences,
                request.similarity_threshold
            )
            sequences.extend(similar_sequences)
            sequences = list(set(sequences))  # Remove duplicates

        # Rank sequences using workflow
        result = workflow.rank_sequences(sequences, request.requirements.dict())

        if result.get('success'):
            ranked_sequences = result.get('ranked_sequences', [])

            # Apply user requirements filtering
            filtered_sequences = apply_user_requirements(
                ranked_sequences,
                request.requirements
            )

            return AnalysisResponse(
                success=True,
                message=f"Successfully ranked {len(filtered_sequences)} sequences",
                data={
                    "total_sequences": len(sequences),
                    "ranked_sequences": filtered_sequences,
                    "requirements_applied": request.requirements.dict(),
                    "database_matches_included": request.include_database_matches
                }
            )
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Ranking failed'))

    except Exception as e:
        logger.error(f"Sequence ranking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def find_similar_sequences(query_sequences: List[str], threshold: float = 0.8) -> List[str]:
    """Find similar sequences from database"""
    if sequence_database.empty:
        return []

    try:
        embedder = SequenceEmbedder()
        similar_sequences = []

        for query_seq in query_sequences:
            query_embedding = embedder.embed_sequence(query_seq)

            # Simple similarity search (can be optimized with vector database)
            for _, row in sequence_database.iterrows():
                db_seq = row['sequence']
                if db_seq not in query_sequences:  # Avoid duplicates
                    db_embedding = embedder.embed_sequence(db_seq)
                    similarity = np.dot(query_embedding, db_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
                    )

                    if similarity >= threshold:
                        similar_sequences.append(db_seq)

        return similar_sequences[:10]  # Limit to top 10 matches

    except Exception as e:
        logger.error(f"Error finding similar sequences: {e}")
        return []

def apply_user_requirements(ranked_sequences: List[Dict], requirements: UserRequirements) -> List[Dict]:
    """Apply user requirements to filter and re-rank sequences"""
    filtered = ranked_sequences.copy()

    # Apply sequence length filter
    if requirements.sequence_length_range:
        min_len, max_len = requirements.sequence_length_range
        filtered = [seq for seq in filtered
                   if min_len <= len(seq['sequence']) <= max_len]

    # Apply MIC range filter (if available in properties)
    if requirements.mic_range:
        min_mic, max_mic = requirements.mic_range
        # This would need actual MIC data from database
        # For now, we'll use a placeholder scoring
        for seq in filtered:
            # Estimate MIC based on sequence properties
            estimated_mic = estimate_mic_from_sequence(seq['sequence'])
            seq['estimated_mic'] = estimated_mic
            seq['meets_mic_requirement'] = min_mic <= estimated_mic <= max_mic

    # Sort by user requirements compliance
    filtered.sort(key=lambda x: (
        x.get('meets_mic_requirement', True),
        -x.get('score', 0)
    ), reverse=True)

    return filtered

def estimate_mic_from_sequence(sequence: str) -> float:
    """Estimate MIC value based on sequence properties"""
    # Simple heuristic based on charge and hydrophobicity
    charge = sum(1 for aa in sequence if aa in 'KRH') - sum(1 for aa in sequence if aa in 'DE')
    hydrophobic_ratio = sum(1 for aa in sequence if aa in 'AILMFPWV') / len(sequence)

    # Lower MIC (better activity) for higher charge and moderate hydrophobicity
    estimated_mic = max(0.5, 10.0 - (charge * 0.5) - (hydrophobic_ratio * 5.0))
    return round(estimated_mic, 2)

# Training endpoints
@app.post("/training/run", response_model=TrainingResponse)
async def run_training_session(request: TrainingRequest):
    """Run a training session to accumulate experience from sequence comparisons"""
    try:
        if not training_module:
            raise HTTPException(status_code=500, detail="Training module not initialized")

        logger.info(f"Starting training session with {request.max_pairs} max pairs")

        # Update training configuration if provided
        if request.similarity_threshold:
            training_module.config.similarity_threshold = request.similarity_threshold
        if request.batch_size:
            training_module.config.batch_size = request.batch_size

        # Run training session
        result = training_module.run_training_session(max_pairs=request.max_pairs)

        if result.get("success"):
            return TrainingResponse(
                success=True,
                message=f"Training session completed successfully",
                session_stats=result.get("session_stats"),
                overall_stats=result.get("overall_stats")
            )
        else:
            return TrainingResponse(
                success=False,
                message="Training session failed",
                error=result.get("error")
            )

    except Exception as e:
        logger.error(f"Training session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/stats")
async def get_training_statistics():
    """Get training statistics and experience database information"""
    try:
        if not training_module:
            raise HTTPException(status_code=500, detail="Training module not initialized")

        stats = training_module.get_training_statistics()
        return {
            "success": True,
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Failed to get training statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/reset")
async def reset_training_statistics():
    """Reset training statistics"""
    try:
        if not training_module:
            raise HTTPException(status_code=500, detail="Training module not initialized")

        training_module.reset_training_stats()
        return {
            "success": True,
            "message": "Training statistics reset successfully"
        }

    except Exception as e:
        logger.error(f"Failed to reset training statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/export")
async def export_training_data(filepath: str = "training_export.json"):
    """Export training data and experiences to file"""
    try:
        if not training_module:
            raise HTTPException(status_code=500, detail="Training module not initialized")

        # Ensure safe file path
        safe_filepath = os.path.join(DATA_DIR, os.path.basename(filepath))

        success = training_module.export_training_data(safe_filepath)

        if success:
            return {
                "success": True,
                "message": f"Training data exported to {safe_filepath}",
                "filepath": safe_filepath
            }
        else:
            return {
                "success": False,
                "message": "Failed to export training data"
            }

    except Exception as e:
        logger.error(f"Failed to export training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def extract_sequences_from_message(message: str) -> List[str]:
    """Extract peptide sequences from user message"""
    import re

    # Pattern to match peptide sequences (uppercase letters, 8-50 characters)
    sequence_pattern = r'\b[ACDEFGHIKLMNPQRSTVWY]{8,50}\b'
    sequences = re.findall(sequence_pattern, message.upper())

    # Remove duplicates while preserving order
    seen = set()
    unique_sequences = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            unique_sequences.append(seq)

    return unique_sequences


async def perform_comprehensive_ranking_analysis(sequences: List[str], user_query: str, session_id: str) -> Dict[str, Any]:
    """
    Perform comprehensive ranking analysis with LLM guidance following the correct workflow:
    1. Get experiences from experience database
    2. Analyze each sequence individually with LLM guidance
    3. Perform comprehensive comparison and ranking
    4. Generate detailed analysis report
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        from configuration import OPENAI_API_KEY, CHAT_MODEL

        # Initialize LLM
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )

        analysis_steps = []

        # Step 1: Get relevant experiences from database
        logger.info("Step 1: Retrieving relevant experiences from database...")
        relevant_experiences = experience_manager.get_relevant_experiences(
            query=f"antimicrobial peptide ranking analysis {user_query}",
            top_k=15
        )

        analysis_steps.append({
            "step": "experience_retrieval",
            "description": f"Retrieved {len(relevant_experiences)} relevant experiences from database",
            "experiences_count": len(relevant_experiences)
        })

        # Step 2: Individual sequence analysis with LLM guidance
        logger.info("Step 2: Performing individual sequence analysis...")
        individual_analyses = []

        for i, sequence in enumerate(sequences):
            logger.info(f"Analyzing sequence {i+1}/{len(sequences)}: {sequence[:20]}...")

            # Calculate basic properties
            properties = calculate_sequence_properties(sequence)

            # Get sequence-specific experiences
            seq_experiences = []
            for exp in relevant_experiences:
                # Handle both Experience objects and dictionaries
                if hasattr(exp, 'content'):
                    content = exp.content.upper()
                else:
                    content = exp.get('content', '').upper()

                if any(seq_word in content for seq_word in [sequence[:10], sequence[-10:]]):
                    seq_experiences.append(exp)

            # LLM-guided individual analysis
            individual_prompt = f"""
## Individual Antimicrobial Peptide Sequence Analysis

**Sequence to Analyze**: {sequence}
**Length**: {len(sequence)} amino acids
**Basic Properties**:
- Net Charge: {properties['net_charge']}
- Hydrophobic Ratio: {properties['hydrophobic_ratio']:.3f}
- Aromatic Ratio: {properties['aromatic_ratio']:.3f}

**Relevant Experience Context** ({len(seq_experiences)} experiences):
{format_experience_context(seq_experiences[:5])}

**Analysis Requirements**:
1. **Structural Analysis**: Analyze the sequence composition, charge distribution, and amphipathic potential
2. **Antimicrobial Potential**: Predict MIC range and target bacteria based on sequence features
3. **Mechanism Prediction**: Predict likely mechanism of action (membrane disruption, intracellular targets, etc.)
4. **Safety Assessment**: Evaluate potential toxicity and selectivity concerns
5. **Synthesis Feasibility**: Assess ease of synthesis and potential modifications

**Output Format**: Provide detailed analysis in JSON format with scores (0-100) for each aspect.
"""

            messages = [
                SystemMessage(content="""You are a professional antimicrobial peptide design expert. Analyze each sequence individually using accumulated experience and scientific principles. Provide detailed, evidence-based analysis with quantitative predictions."""),
                HumanMessage(content=individual_prompt)
            ]

            try:
                response = llm.invoke(messages)
                individual_analysis = {
                    "sequence": sequence,
                    "analysis": response.content,
                    "properties": properties,
                    "experience_support": len(seq_experiences)
                }
                individual_analyses.append(individual_analysis)

            except Exception as e:
                logger.error(f"Individual analysis failed for sequence {i+1}: {e}")
                individual_analyses.append({
                    "sequence": sequence,
                    "analysis": f"Analysis failed: {str(e)}",
                    "properties": properties,
                    "experience_support": 0
                })

        analysis_steps.append({
            "step": "individual_analysis",
            "description": f"Completed individual analysis for {len(sequences)} sequences",
            "sequences_analyzed": len(individual_analyses)
        })

        # Step 3: Comprehensive comparison and ranking with LLM guidance
        logger.info("Step 3: Performing comprehensive comparison and ranking...")

        # Prepare comprehensive ranking prompt
        ranking_prompt = f"""
## Comprehensive Antimicrobial Peptide Ranking Analysis

**User Query**: {user_query}
**Number of Sequences**: {len(sequences)}

**Individual Analysis Results**:
{format_individual_analyses(individual_analyses)}

**Experience Database Context** ({len(relevant_experiences)} total experiences):
{format_experience_context(relevant_experiences[:10])}

**Ranking Task**:
Based on the individual analyses and accumulated experience, rank these {len(sequences)} sequences from most promising to least promising for antimicrobial applications.

**Evaluation Criteria** (in order of importance):
1. **Antimicrobial Potency** (30%): Predicted MIC values and broad-spectrum activity
2. **Bacterial Selectivity** (25%): Preference for bacterial vs mammalian cells
3. **Structural Stability** (20%): Resistance to proteases and environmental factors
4. **Safety Profile** (15%): Low toxicity and hemolytic activity
5. **Synthesis Feasibility** (10%): Cost and complexity of production

**Required Output**:
1. **Complete Ranking**: Rank ALL {len(sequences)} sequences (no partial rankings)
2. **Detailed Rationale**: Explain ranking decisions with scientific evidence
3. **Comparative Analysis**: Highlight key differences between sequences
4. **Optimization Suggestions**: Provide improvement recommendations for top candidates
5. **Confidence Assessment**: Rate confidence in each ranking decision

**Output Format**: Provide comprehensive analysis in structured format with clear ranking and detailed explanations.
"""

        messages = [
            SystemMessage(content="""You are a senior antimicrobial peptide researcher with 20+ years of experience in drug design and clinical development. Provide comprehensive, evidence-based ranking with detailed scientific rationale. Your analysis will guide important research decisions."""),
            HumanMessage(content=ranking_prompt)
        ]

        try:
            ranking_response = llm.invoke(messages)
            comprehensive_analysis = ranking_response.content

            analysis_steps.append({
                "step": "comprehensive_ranking",
                "description": "Completed comprehensive comparison and ranking analysis",
                "ranking_completed": True
            })

        except Exception as e:
            logger.error(f"Comprehensive ranking failed: {e}")
            comprehensive_analysis = f"Comprehensive ranking analysis failed: {str(e)}"
            analysis_steps.append({
                "step": "comprehensive_ranking",
                "description": "Comprehensive ranking failed",
                "error": str(e)
            })

        # Step 4: Generate final detailed report
        logger.info("Step 4: Generating final analysis report...")

        final_report = f"""# Comprehensive Antimicrobial Peptide Ranking Analysis

## Analysis Overview
- **Sequences Analyzed**: {len(sequences)}
- **Experience Database**: {len(relevant_experiences)} relevant experiences utilized
- **Analysis Method**: Individual sequence analysis followed by comprehensive comparison

## Individual Sequence Analyses
{format_individual_analyses_summary(individual_analyses)}

## Comprehensive Ranking and Comparison
{comprehensive_analysis}

## Analysis Methodology
1. **Experience Retrieval**: Retrieved {len(relevant_experiences)} relevant experiences from accumulated knowledge base
2. **Individual Analysis**: Each sequence analyzed independently using LLM guidance and experience context
3. **Comprehensive Comparison**: All sequences compared systematically across multiple criteria
4. **Evidence-Based Ranking**: Final ranking based on scientific evidence and professional expertise

## Key Insights and Recommendations
Based on this comprehensive analysis, the ranking provides actionable insights for antimicrobial peptide development and optimization.

---
*Analysis completed using {len(relevant_experiences)} database experiences and professional AMP design expertise.*
"""

        analysis_steps.append({
            "step": "final_report",
            "description": "Generated comprehensive analysis report",
            "report_length": len(final_report)
        })

        return {
            "success": True,
            "response": final_report,
            "analysis_steps": analysis_steps,
            "sequences_count": len(sequences),
            "experiences_used": len(relevant_experiences)
        }

    except Exception as e:
        logger.error(f"Comprehensive ranking analysis failed: {e}")
        return {
            "success": False,
            "response": f"Sorry, the comprehensive analysis failed: {str(e)}",
            "analysis_steps": analysis_steps,
            "error": str(e)
        }


def calculate_sequence_properties(sequence: str) -> Dict[str, Any]:
    """Calculate basic sequence properties"""
    length = len(sequence)

    # Amino acid classifications
    positive_aa = set('KRH')
    negative_aa = set('DE')
    hydrophobic_aa = set('AILMFPWV')
    aromatic_aa = set('FWY')

    # Calculate properties
    positive_count = sum(1 for aa in sequence if aa in positive_aa)
    negative_count = sum(1 for aa in sequence if aa in negative_aa)
    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
    aromatic_count = sum(1 for aa in sequence if aa in aromatic_aa)

    return {
        "length": length,
        "net_charge": positive_count - negative_count,
        "positive_charge": positive_count,
        "negative_charge": negative_count,
        "hydrophobic_ratio": hydrophobic_count / length if length > 0 else 0,
        "aromatic_ratio": aromatic_count / length if length > 0 else 0,
        "hydrophobic_count": hydrophobic_count,
        "aromatic_count": aromatic_count
    }


def format_experience_context(experiences: List[Any]) -> str:
    """Format experience context for LLM prompt"""
    if not experiences:
        return "No relevant experiences found."

    formatted = []
    for i, exp in enumerate(experiences[:5], 1):  # Limit to top 5
        # Handle both Experience objects and dictionaries
        if hasattr(exp, 'content'):
            content = exp.content
            confidence = exp.confidence
        else:
            content = exp.get('content', 'No content')
            confidence = exp.get('confidence', 0.0)

        formatted.append(f"{i}. {content[:200]}... (Confidence: {confidence:.2f})")

    return "\n".join(formatted)


def format_individual_analyses(analyses: List[Dict[str, Any]]) -> str:
    """Format individual analyses for comprehensive ranking prompt"""
    formatted = []
    for i, analysis in enumerate(analyses, 1):
        sequence = analysis['sequence']
        properties = analysis['properties']
        analysis_text = analysis['analysis'][:500] + "..." if len(analysis['analysis']) > 500 else analysis['analysis']

        formatted.append(f"""
**Sequence {i}**: {sequence}
- Length: {properties['length']} aa
- Net Charge: {properties['net_charge']}
- Hydrophobic Ratio: {properties['hydrophobic_ratio']:.3f}
- Analysis: {analysis_text}
- Experience Support: {analysis['experience_support']} experiences
""")

    return "\n".join(formatted)


def format_individual_analyses_summary(analyses: List[Dict[str, Any]]) -> str:
    """Format individual analyses summary for final report"""
    formatted = []
    for i, analysis in enumerate(analyses, 1):
        sequence = analysis['sequence']
        properties = analysis['properties']

        formatted.append(f"""
### Sequence {i}: {sequence}
- **Length**: {properties['length']} amino acids
- **Net Charge**: {properties['net_charge']}
- **Hydrophobic Ratio**: {properties['hydrophobic_ratio']:.3f}
- **Aromatic Ratio**: {properties['aromatic_ratio']:.3f}
- **Experience Support**: {analysis['experience_support']} relevant experiences
""")

    return "\n".join(formatted)


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting AMP Designer Agent API Server - Version 1 on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
