"""
Router Agent for AMPilot
Determines user query intent: sequence-to-properties vs properties-to-sequence
"""

from typing import Literal, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import re

class QueryIntent(BaseModel):
    """Structured output for query intent classification."""
    intent: Literal["sequence_to_properties", "properties_to_sequence", "general_conversation"] = Field(
        description="The type of query: sequence_to_properties (user provides sequence, wants properties), properties_to_sequence (user provides properties, wants sequences), or general_conversation"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification decision"
    )
    extracted_info: Dict[str, Any] = Field(
        description="Extracted information from the query (sequence, bacterium, MIC range, modifications, etc.)",
        default_factory=dict
    )

    model_config = {"extra": "forbid"}  # This ensures additionalProperties = false for OpenAI structured output

class RouterAgent:
    """Router agent to classify user queries and extract relevant information."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are a specialized router for an antimicrobial peptide (AMP) research system.
Your job is to analyze user queries and determine their intent:

1. sequence_to_properties: User provides a peptide sequence and wants to know its properties (MIC values, target bacteria, modifications, etc.)
   - Examples: "What is the MIC of GLPRKILCAIAKKKGKCKGPLKLVCKC?", "Tell me about this sequence: GFGCPGDAYQCSEHCRALGGGRTGGYCAGPWYLGHPTCTCSF", "What is the principle of KLAKLAKKLAKLAK?"

2. properties_to_sequence: User provides desired properties and wants to find matching peptide sequences. This includes:
   - Target bacteria (e.g., "active against S. aureus", "target C. albicans")
   - MIC values (e.g., "MIC < 10 µM", "around 13 MIC value")
   - Mechanisms (e.g., "membrane disruption", "pore formation", "cell wall synthesis inhibition")
   - Modifications (e.g., "disulfide bonds", "amidation")
   - Other properties (e.g., "cationic peptides", "amphipathic")
   - Examples: "Find peptides active against S. aureus", "Are there any AMPs that have the principle of membrane disruption?", "Show me peptides with MIC < 10 µM", "Find modified peptides with disulfide bonds", "are there amps have property of target C. albicans and have around 13 mic value"

3. general_conversation: General questions, greetings, or non-specific AMP queries
   - Examples: "Hello", "What are antimicrobial peptides?", "How do AMPs work?", "what is the weather today?"

Important tie-break rule:
- If the query contains an explicit peptide sequence (uppercase amino acid string, typically 10+ letters), ALWAYS classify as sequence_to_properties, even if property/bacteria/mechanism keywords are also present.

Extract relevant information:
- Sequences: Any amino acid sequences (usually uppercase letters like GLPRKILCAIAKKKGKCKGPLKLVCKC)
- Bacteria: Bacterial names (e.g., S. aureus, E. coli, B. subtilis, C. albicans)
- MIC values: Concentration ranges (e.g., "< 10 µM", "between 1-5 µM", "around 13")
- Mechanisms: Action mechanisms (e.g., "membrane disruption", "pore formation")
- Modifications: Disulfide bonds, amidation, etc.
- Strains: Specific bacterial strains (e.g., ATCC29213)

Be precise in your classification and extract as much relevant information as possible."""

    def classify_query(self, user_query: str) -> QueryIntent:
        """Classify user query and extract relevant information."""
        
        # Create structured LLM with function calling to avoid schema issues
        structured_llm = self.llm.with_structured_output(QueryIntent, method="function_calling")
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Classify this query: {user_query}")
        ]
        
        try:
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            print(f"Router classification error: {e}")
            # Fallback classification
            return self._fallback_classification(user_query)
    
    def _fallback_classification(self, user_query: str) -> QueryIntent:
        """Fallback classification using simple heuristics."""
        query_lower = user_query.lower()
        
        # Check for amino acid sequences (uppercase letters, typically 10+ chars)
        sequence_pattern = r'[ACDEFGHIKLMNPQRSTVWY]{10,}'
        has_sequence = bool(re.search(sequence_pattern, user_query))
        
        # Check for property-related keywords
        property_keywords = ['mic', 'bacteria', 'strain', 'modified', 'disulfide', 'amidation', 'active against',
                           'membrane disruption', 'pore formation', 'target', 'principle', 'mechanism',
                           'cationic', 'amphipathic', 'antimicrobial activity', 'c. albicans', 's. aureus',
                           'e. coli', 'candida', 'find', 'show me', 'are there']
        has_properties = any(keyword in query_lower for keyword in property_keywords)

        # Check for bacterial names
        bacteria_pattern = r'[A-Z]\.\s*[a-z]+|[A-Z][a-z]+\s+[a-z]+|candida|albicans|aureus|coli'
        has_bacteria = bool(re.search(bacteria_pattern, user_query, re.IGNORECASE))

        # Check for mechanism-related queries
        mechanism_keywords = ['principle', 'mechanism', 'membrane disruption', 'pore formation', 'how do', 'function']
        has_mechanism = any(keyword in query_lower for keyword in mechanism_keywords)

        # Prefer sequence_to_properties whenever a valid sequence is present
        if has_sequence:
            intent = "sequence_to_properties"
            # Slightly higher confidence if properties are also present
            confidence = 0.85 if (has_properties or has_bacteria or has_mechanism) else 0.8
            reasoning = "Detected an amino acid sequence; sequence takes precedence over property keywords"
        elif (has_properties or has_bacteria or has_mechanism):
            intent = "properties_to_sequence"
            confidence = 0.7
            reasoning = "Detected property-related keywords, bacterial names, or mechanism queries"
        else:
            intent = "general_conversation"
            confidence = 0.6
            reasoning = "General query or unclear intent"

        return QueryIntent(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            extracted_info={"query": user_query}
        )

@tool
def route_query(query: str) -> Dict[str, Any]:
    """
    Route user query to determine intent and extract relevant information.
    
    Args:
        query: User's natural language query about antimicrobial peptides
        
    Returns:
        Dictionary containing intent classification and extracted information
    """
    try:
        from .configuration import OPENAI_API_KEY
    except ImportError:
        # Fallback for when called from different contexts
        from configuration import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        # Use robust fallback routing when API key is missing
        router = RouterAgent(llm=None)  # llm is not used in fallback
        result = router._fallback_classification(query)
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": f"Fallback routing (no API key): {result.reasoning}",
            "extracted_info": result.extracted_info
        }

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )

        router = RouterAgent(llm)
        result = router.classify_query(query)

        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "extracted_info": result.extracted_info
        }

    except Exception as e:
        import traceback
        print("Router error:\n", traceback.format_exc())
        return {
            "error": f"Router failed: {str(e)}",
            "intent": "general_conversation",
            "confidence": 0.0
        }

# Test function
def test_router():
    """Test the router with sample queries."""
    test_queries = [
        "What is the MIC of GLPRKILCAIAKKKGKCKGPLKLVCKC?",
        "Find peptides active against S. aureus with MIC < 10 µM",
        "Hello, what are antimicrobial peptides?",
        "Show me modified peptides with disulfide bonds",
        "GFGCPGDAYQCSEHCRALGGGRTGGYCAGPWYLGHPTCTCSF properties"
    ]
    
    for query in test_queries:
        result = route_query(query)
        print(f"Query: {query}")
        print(f"Intent: {result.get('intent')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Reasoning: {result.get('reasoning')}")
        print("-" * 50)

if __name__ == "__main__":
    test_router()
