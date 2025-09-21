"""
Semantic embedding-based sequence clustering module

Uses BAAI/bge-large-en-v1.5 model for semantic encoding of antimicrobial peptide sequences,
then performs clustering based on cosine similarity to generate CSV files of similar sequence pairs.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

from tools.sequence_embedder import SequenceEmbedder, SequenceClusterer
from configuration import (
    GRAMPA_DATA_FILE, CLUSTERS_DIR, MAX_CLUSTER_FILES,
    SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceDataProcessor:
    """Sequence data processor"""

    def __init__(self):
        """Initialize data processor - using cosine similarity for sequence alignment"""
        self.embedder = SequenceEmbedder()
        self.clusterer = SequenceClusterer(self.embedder)
        logger.info("Sequence data processor initialized (using cosine similarity alignment)")

    def load_data(self, filepath: str = GRAMPA_DATA_FILE) -> pd.DataFrame:
        """
        Load antimicrobial peptide data

        Args:
            filepath: Data file path

        Returns:
            Data DataFrame
        """
        logger.info(f"Loading data file: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data loading completed, total rows: {len(df)}")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Preprocess data, remove duplicates and filter

        Args:
            df: Original data DataFrame

        Returns:
            Unique sequence list and corresponding data
        """
        logger.info("Preprocessing data...")

        # Remove null values
        df_clean = df.dropna(subset=['sequence'])

        # Get unique sequences
        unique_sequences = df_clean['sequence'].unique().tolist()
        logger.info(f"Number of unique sequences: {len(unique_sequences)}")

        # Filter sequence length (5-100 amino acids)
        filtered_sequences = []
        for seq in unique_sequences:
            if 5 <= len(seq) <= 100:
                filtered_sequences.append(seq)

        logger.info(f"Number of sequences after filtering: {len(filtered_sequences)}")

        # If too many sequences, sample them
        if len(filtered_sequences) > 2000:
            logger.info("Too many sequences, randomly sampling 2000 for analysis...")
            np.random.seed(42)
            sample_indices = np.random.choice(len(filtered_sequences), 2000, replace=False)
            filtered_sequences = [filtered_sequences[i] for i in sample_indices]

        # Create sequence to data mapping
        sequence_data_map = {}
        for seq in filtered_sequences:
            sequence_data_map[seq] = df_clean[df_clean['sequence'] == seq]

        return filtered_sequences, sequence_data_map

    def find_similar_pairs_and_save(self, sequences: List[str], sequence_data_map: Dict[str, pd.DataFrame]) -> int:
        """
        Find similar sequence pairs and save as CSV files

        Args:
            sequences: Sequence list
            sequence_data_map: Sequence to data mapping

        Returns:
            Number of saved files
        """
        logger.info("Finding similar sequence pairs using cosine similarity...")

        # Find similar sequence pairs
        similar_pairs = self.clusterer.find_similar_pairs(sequences, SIMILARITY_THRESHOLD)

        logger.info(f"Found {len(similar_pairs)} similar sequence pairs")

        # Save similar sequence pairs
        saved_files = 0
        processed_sequences = set()

        for i, (idx1, idx2, similarity) in enumerate(similar_pairs):
            if MAX_CLUSTER_FILES is not None and saved_files >= MAX_CLUSTER_FILES:
                break

            seq1, seq2 = sequences[idx1], sequences[idx2]

            # Avoid processing the same sequences repeatedly
            pair_key = tuple(sorted([seq1, seq2]))
            if pair_key in processed_sequences:
                continue
            processed_sequences.add(pair_key)

            # Get all data for both sequences
            data1 = sequence_data_map[seq1]
            data2 = sequence_data_map[seq2]

            # Ensure data is in DataFrame format
            if not isinstance(data1, pd.DataFrame):
                continue
            if not isinstance(data2, pd.DataFrame):
                continue

            # Merge data
            combined_data = pd.concat([data1, data2], ignore_index=True)

            # Add similarity information
            combined_data['pair_similarity'] = similarity
            combined_data['pair_id'] = f"pair_{saved_files + 1}"

            # Save file
            filename = f"similar_pair_{saved_files + 1}_sim_{similarity:.3f}.csv"
            filepath = os.path.join(CLUSTERS_DIR, filename)
            combined_data.to_csv(filepath, index=False)

            logger.info(f"Saved similar sequence pair {saved_files + 1}: {filename}")
            logger.info(f"  Sequence 1: {seq1[:30]}...")
            logger.info(f"  Sequence 2: {seq2[:30]}...")
            logger.info(f"  Similarity: {similarity:.3f}")
            logger.info(f"  Data rows: {len(combined_data)}")

            saved_files += 1

        return saved_files

    def cluster_and_save(self, sequences: List[str], sequence_data_map: Dict[str, pd.DataFrame]) -> int:
        """
        Cluster sequences and save results

        Args:
            sequences: Sequence list
            sequence_data_map: Sequence to data mapping

        Returns:
            Number of saved cluster files
        """
        logger.info("Performing sequence clustering...")

        # Perform clustering
        cluster_labels, embeddings = self.clusterer.cluster_sequences(sequences)

        # Analyze clustering results
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise points

        logger.info(f"Found {len(unique_labels)} clusters")
        logger.info(f"Number of noise points: {sum(cluster_labels == -1)}")

        # Save clustering results
        saved_files = 0
        cluster_info = []

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_sequences = [sequences[i] for i in cluster_indices]

            # Collect data for all sequences in the cluster
            cluster_data_list = []
            for seq in cluster_sequences:
                cluster_data_list.append(sequence_data_map[seq])

            if cluster_data_list:
                cluster_data = pd.concat(cluster_data_list, ignore_index=True)
                cluster_data['cluster_id'] = label

                cluster_info.append({
                    'cluster_id': label,
                    'size': len(cluster_data),
                    'sequences': cluster_sequences,
                    'data': cluster_data
                })

        # Sort by cluster size
        cluster_info.sort(key=lambda x: x['size'], reverse=True)

        # Save cluster files
        for cluster in cluster_info:
            if MAX_CLUSTER_FILES is not None and saved_files >= MAX_CLUSTER_FILES:
                break

            cluster_id = cluster['cluster_id']
            cluster_data = cluster['data']

            filename = f"cluster_{cluster_id}_size_{len(cluster_data)}.csv"
            filepath = os.path.join(CLUSTERS_DIR, filename)
            cluster_data.to_csv(filepath, index=False)

            logger.info(f"Saved cluster {cluster_id}: {filename}, containing {len(cluster_data)} records")
            saved_files += 1

        return saved_files

def main():
    """Main function: Execute sequence clustering and similar sequence pair generation"""
    logger.info("Starting sequence clustering analysis...")

    # Initialize data processor
    processor = SequenceDataProcessor()

    # Load embedding cache
    cache_file = os.path.join(CLUSTERS_DIR, "embeddings_cache.pkl")
    processor.embedder.load_embeddings_cache(cache_file)

    try:
        # Load and preprocess data
        df = processor.load_data()
        sequences, sequence_data_map = processor.preprocess_data(df)

        logger.info(f"Starting analysis of {len(sequences)} sequences")

        # Method 1: Find similar sequence pairs based on similarity threshold
        logger.info("=" * 50)
        logger.info("Method 1: Finding similar sequence pairs based on similarity threshold")
        logger.info("=" * 50)

        pairs_saved = processor.find_similar_pairs_and_save(sequences, sequence_data_map)

        # Method 2: Clustering analysis (if similar sequence pairs are insufficient)
        if MAX_CLUSTER_FILES is not None and pairs_saved < MAX_CLUSTER_FILES // 2:
            logger.info("=" * 50)
            logger.info("Method 2: Supplementary clustering analysis")
            logger.info("=" * 50)

            clusters_saved = processor.cluster_and_save(sequences, sequence_data_map)
            total_saved = pairs_saved + clusters_saved
        else:
            total_saved = pairs_saved

        # Save embedding cache
        processor.embedder.save_embeddings_cache(cache_file)

        logger.info("=" * 50)
        logger.info("Sequence clustering analysis completed!")
        logger.info(f"Total of {total_saved} files saved to {CLUSTERS_DIR}")
        logger.info("=" * 50)

        # Display list of saved files
        saved_files = [f for f in os.listdir(CLUSTERS_DIR) if f.endswith('.csv')]
        logger.info(f"List of saved files:")
        for i, filename in enumerate(saved_files[:10], 1):  # Show first 10 files
            logger.info(f"  {i}. {filename}")
        if len(saved_files) > 10:
            logger.info(f"  ... and {len(saved_files) - 10} more files")

    except Exception as e:
        logger.error(f"Error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
