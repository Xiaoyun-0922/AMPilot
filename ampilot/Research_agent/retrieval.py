import weaviate
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
try:
    from .configuration import WEAVIATE_URL, WEAVIATE_COLLECTION_NAME
    from .embeddings import get_embedding_model
except ImportError:
    # Fallback for when called from different contexts
    from configuration import WEAVIATE_URL, WEAVIATE_COLLECTION_NAME
    from embeddings import get_embedding_model

try:
    weaviate_client = weaviate.connect_to_local(
        host=WEAVIATE_URL.replace("http://", "").split(":")[0],
        port=int(WEAVIATE_URL.replace("http://", "").split(":")[1])
    )
    amp_collection = weaviate_client.collections.get(WEAVIATE_COLLECTION_NAME)
    print("Successfully connected to Weaviate and retrieved GRAMPA collection.")
    embedding_model = get_embedding_model()

except Exception as e:
    print(f"Failed to initialize Weaviate client or embedding model: {e}")
    weaviate_client = None
    embedding_model = None



@tool
def find_peptide_properties(sequence: str) -> List[Dict[str, Any]]:
    """
    Find antimicrobial properties for a given peptide sequence.

    This tool searches the GRAMPA database to find MIC values, target bacteria,
    and modification information for a specific peptide sequence.

    Args:
        sequence: Amino acid sequence (e.g., "GLPRKILCAIAKKKGKCKGPLKLVCKC")

    Returns:
        List of dictionaries containing:
        - sequence: The peptide sequence
        - bacterium: Target bacterium
        - strain: Bacterial strain (if available)
        - mic_value_um: MIC value in µM
        - modifications: Peptide modifications
        - database: Source database
    """
    if not weaviate_client or not embedding_model:
        return [{"error": "Database not available. Please ensure Weaviate is running."}]

    try:
        # Search for exact sequence matches first
        response = amp_collection.query.bm25(
            query=sequence,
            limit=10,
            return_properties=None
        )

        exact_matches = []
        for item in response.objects:
            if item.properties.get('sequence', '').upper() == sequence.upper():
                exact_matches.append(item.properties)

        if exact_matches:
            return exact_matches

        # Strict exact matching only: if no exact match, inform user
        return [{"message": f"No properties found for sequence: {sequence}. The sequence was not found in the GRAMPA database."}]

    except Exception as e:
        print(f"Error in find_peptide_properties: {e}")
        return [{"error": "Failed to search for peptide properties."}]


@tool
def find_peptides_by_properties(bacterium: str = "", mic_range: str = "", modifications: str = "", strain: str = "") -> List[Dict[str, Any]]:
    """
    Find peptide sequences based on desired antimicrobial properties.

    This tool searches the GRAMPA database to find peptides that match specified criteria
    such as target bacterium, MIC range, modifications, or bacterial strain.

    Args:
        bacterium: Target bacterium name (e.g., "S. aureus", "E. coli")
        mic_range: MIC concentration range (e.g., "< 10", "1-5", "> 100")
        modifications: Type of modifications (e.g., "disulfide", "amidation", "none")
        strain: Specific bacterial strain (e.g., "ATCC29213")

    Returns:
        List of dictionaries containing matching peptide information
    """
    if not weaviate_client or not embedding_model:
        return [{"error": "Database not available. Please ensure Weaviate is running."}]

    try:
        # Build search query based on provided criteria
        search_terms = []
        if bacterium:
            search_terms.append(f"bacterium: {bacterium}")
        if strain:
            search_terms.append(f"strain: {strain}")
        if modifications:
            search_terms.append(f"modifications: {modifications}")
        if mic_range:
            search_terms.append(f"MIC: {mic_range}")

        if not search_terms:
            return [{"message": "Please provide at least one search criterion (bacterium, MIC range, modifications, or strain)."}]

        query_text = " | ".join(search_terms)

        # Use semantic search to find matching peptides
        query_vector = embedding_model.embed_query(query_text)

        response = amp_collection.query.near_vector(
            near_vector=query_vector,
            limit=10,
            return_properties=None
        )

        results = []
        for item in response.objects:
            result = item.properties.copy()

            # Apply additional filtering based on criteria
            if bacterium and bacterium.lower() not in result.get('bacterium', '').lower():
                continue
            if strain and strain.lower() not in result.get('strain', '').lower():
                continue

            # Parse MIC range if provided
            if mic_range:
                mic_value = result.get('mic_value_um', 0)
                if not _matches_mic_range(mic_value, mic_range):
                    continue

            results.append(result)

        if not results:
            return [{"message": f"No peptides found matching the criteria: {query_text}"}]

        return results[:5]  # Limit to top 5 results for concise presentation

    except Exception as e:
        print(f"Error in find_peptides_by_properties: {e}")
        return [{"error": "Failed to search for peptides by properties."}]


def _matches_mic_range(mic_value: float, mic_range: str) -> bool:
    """Helper function to check if MIC value matches the specified range."""
    try:
        mic_range = mic_range.strip().lower()

        if mic_range.startswith('<'):
            threshold = float(mic_range[1:].strip())
            return mic_value < threshold
        elif mic_range.startswith('>'):
            threshold = float(mic_range[1:].strip())
            return mic_value > threshold
        elif '-' in mic_range:
            parts = mic_range.split('-')
            if len(parts) == 2:
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                return min_val <= mic_value <= max_val
        else:
            # Exact value
            target = float(mic_range)
            return abs(mic_value - target) < 0.1

    except (ValueError, IndexError):
        return True  # If parsing fails, include the result

    return True

# Note: We no longer need to close the client here as it's a long-lived connection.
# The connection will be closed when the application shuts down.