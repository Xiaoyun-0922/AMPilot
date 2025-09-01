from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_embedding_model() -> Embeddings:
    """
    Initializes a local, high-performance BGE embedding model using the stable,
    generic HuggingFaceEmbeddings class, configured to run on a CUDA-enabled GPU.
    """
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}

    # We use the generic HuggingFaceEmbeddings class, which is the stable foundation
    # for all sentence-transformer models in LangChain.
    # This avoids any issues with renamed or deprecated specialized classes.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    print(
        f"Local BGE embedding model '{model_name}' loaded successfully on CUDA using HuggingFaceEmbeddings."
    )
    return embeddings
