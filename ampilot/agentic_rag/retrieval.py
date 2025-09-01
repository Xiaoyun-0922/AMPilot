from typing import Any, Dict, List

import weaviate
from configuration import WEAVIATE_COLLECTION_NAME, WEAVIATE_URL
from embeddings import get_embedding_model
from langchain_core.tools import tool

try:
    weaviate_client = weaviate.connect_to_local(
        host=WEAVIATE_URL.replace("http://", "").split(":")[0],
        port=int(WEAVIATE_URL.replace("http://", "").split(":")[1]),
    )
    amp_collection = weaviate_client.collections.get(WEAVIATE_COLLECTION_NAME)
    print("Successfully connected to Weaviate and retrieved collection.")
    embedding_model = get_embedding_model()

except Exception as e:
    print(f"Failed to initialize Weaviate client or embedding model: {e}")
    weaviate_client = None
    embedding_model = None


@tool
def retrieve_amp_info(query: str) -> List[Dict[str, Any]]:
    """
    Retrieve relevant structured information from the
    antimicrobial peptide (AMP) database based on user inquiries.

    This function performs semantic search using vector similarity to find
    the most relevant AMP records for a given query.

    Args:
        query: Natural language query about AMPs (e.g., "peptides with antifungal activity")

    Returns:
        List of dictionaries containing AMP information, or an error message if retrieval fails.
    """
    if not weaviate_client or not embedding_model:
        return [
            {
                "error": "Database client or embedding model not initialized. Check server logs."
            }
        ]

    try:
        # Convert query to vector representation using the pre-loaded model
        query_vector = embedding_model.embed_query(query)

        # Perform vector similarity search using the pre-configured collection
        response = amp_collection.query.near_vector(
            near_vector=query_vector,
            limit=3,  # Return top 3 most similar results
            return_properties=None,  # Return all available properties
        )

        # Extract properties from search results
        results = [item.properties for item in response.objects]

        if not results:
            return [
                {
                    "message": "No relevant information found in the database for your query."
                }
            ]

        return results

    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return [{"error": "Failed to retrieve data from the database."}]


# Note: We no longer need to close the client here as it's a long-lived connection.
# The connection will be closed when the application shuts down.
