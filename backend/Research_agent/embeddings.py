from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

def get_embedding_model() -> Embeddings:
    """
    Initializes a local, high-performance BGE embedding model using the stable,
    generic HuggingFaceEmbeddings class, with automatic device detection.
    """
    import torch

    model_name = "BAAI/bge-large-en-v1.5"

    # Detect available device
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available, using GPU for embeddings.")
    else:
        device = "cpu"
        print("CUDA not available, using CPU for embeddings.")

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 32}  # Add batch size control

    try:
        # We use the generic HuggingFaceEmbeddings class, which is the stable foundation
        # for all sentence-transformer models in LangChain.
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        print(f"Local BGE embedding model '{model_name}' loaded successfully on {device.upper()}.")
        return embeddings

    except Exception as e:
        print(f"Failed to load BGE model: {e}")
        print("Falling back to a smaller, more compatible model...")

        # Fallback to a smaller model
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=fallback_model,
                model_kwargs={"device": "cpu"},  # Force CPU for fallback
                encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
            )
            print(f"Fallback model '{fallback_model}' loaded successfully on CPU.")
            return embeddings
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
            raise e2