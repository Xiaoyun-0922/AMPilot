"""
Sequence embedding and similarity calculation module

Uses BAAI/bge-large-en-v1.5 model for semantic encoding of antimicrobial peptide sequences
and calculates cosine similarity between sequences.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict
import pickle
import os
import sys
from tqdm import tqdm
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration import EMBEDDING_MODEL, CLUSTERS_DIR, SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceEmbedder:
    """Sequence embedder class"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize sequence embedder

        Args:
            model_name: Embedding model name
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a single sequence

        Args:
            sequence: Single sequence string

        Returns:
            Embedding vector
        """
        return self.embed_sequences([sequence])[0]
        
    def embed_sequences(self, sequences: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Embed encode a list of sequences

        Args:
            sequences: Sequence list
            use_cache: Whether to use cache

        Returns:
            Embedding vector matrix
        """
        if use_cache:
            # Check cache
            new_sequences = []
            cached_embeddings = []

            for seq in sequences:
                if seq in self.embeddings_cache:
                    cached_embeddings.append(self.embeddings_cache[seq])
                else:
                    new_sequences.append(seq)
                    cached_embeddings.append(None)

            if new_sequences:
                logger.info(f"Encoding {len(new_sequences)} new sequences...")
                new_embeddings = self.model.encode(new_sequences, show_progress_bar=True)

                # Update cache
                new_idx = 0
                for i, seq in enumerate(sequences):
                    if cached_embeddings[i] is None:
                        self.embeddings_cache[seq] = new_embeddings[new_idx]
                        cached_embeddings[i] = new_embeddings[new_idx]
                        new_idx += 1

            return np.array(cached_embeddings)
        else:
            logger.info(f"Encoding {len(sequences)} sequences...")
            return self.model.encode(sequences, show_progress_bar=True)
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix of embedding vectors

        Args:
            embeddings: Embedding vector matrix

        Returns:
            Similarity matrix
        """
        logger.info("Calculating similarity matrix...")
        return cosine_similarity(embeddings)

    def save_embeddings_cache(self, filepath: str):
        """Save embedding cache"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        logger.info(f"Embedding cache saved to: {filepath}")

    def load_embeddings_cache(self, filepath: str):
        """Load embedding cache"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            logger.info(f"Embedding cache loaded from {filepath}, containing {len(self.embeddings_cache)} sequences")
        else:
            logger.info("Embedding cache file not found, will calculate from scratch")

class SequenceClusterer:
    """Sequence clusterer class"""

    def __init__(self, embedder: SequenceEmbedder):
        """
        Initialize sequence clusterer

        Args:
            embedder: Sequence embedder instance
        """
        self.embedder = embedder

    def cluster_sequences(self, sequences: List[str], eps: float = 1 - SIMILARITY_THRESHOLD,
                         min_samples: int = MIN_CLUSTER_SIZE) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster sequences

        Args:
            sequences: Sequence list
            eps: DBSCAN eps parameter (distance threshold)
            min_samples: Minimum number of samples

        Returns:
            Cluster labels and embedding vectors
        """
        # Get embedding vectors
        embeddings = self.embedder.embed_sequences(sequences)

        # Calculate distance matrix (1 - cosine similarity)
        similarity_matrix = self.embedder.calculate_similarity_matrix(embeddings)
        distance_matrix = 1 - similarity_matrix

        # Use DBSCAN for clustering
        logger.info("Performing DBSCAN clustering...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)

        return cluster_labels, embeddings

    def find_similar_pairs(self, sequences: List[str],
                          threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[int, int, float]]:
        """
        Find similar sequence pairs

        Args:
            sequences: Sequence list
            threshold: Similarity threshold

        Returns:
            Similar sequence pair list [(idx1, idx2, similarity), ...]
        """
        embeddings = self.embedder.embed_sequences(sequences)
        similarity_matrix = self.embedder.calculate_similarity_matrix(embeddings)

        similar_pairs = []
        n = len(sequences)

        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))

        # Sort by similarity in descending order
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        return similar_pairs
