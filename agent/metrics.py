"""
Defines similarity metrics for comparing peptide sequences.
This module implements the Strategy Pattern for calculating similarity.
"""
import abc
import pandas as pd
import numpy as np

# You may need to install it: pip install python-Levenshtein
import Levenshtein
# You may need to install modlamp: pip install modlamp
from modlamp.descriptors import GlobalDescriptor
# You may need to install transformers and torch: pip install transformers torch
import torch
from transformers import AutoTokenizer, AutoModel
# For vector comparisons
from scipy.spatial.distance import cdist

# --- 1. Strategy Interface for Similarity Calculation ---

class SimilarityMetric(abc.ABC):
    """
    Abstract Base Class (Interface) for all similarity calculation strategies.
    This defines the contract for how similarity is calculated.
    """
    @abc.abstractmethod
    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-processes the dataframe to generate necessary features (e.g., embeddings).
        This is meant to be called once, not during every search.
        """
        pass

    @abc.abstractmethod
    def calculate_scores(self, target_sequence: str, reference_df: pd.DataFrame) -> pd.Series:
        """
        Calculates similarity scores between a target and a dataframe of references.

        Returns:
            A pandas Series containing scores, indexed the same as reference_df.
        """
        pass

    @property
    @abc.abstractmethod
    def sort_ascending(self) -> bool:
        """
        Specifies whether a lower score means more similar (True for distance)
        or a higher score means more similar (False for similarity).
        """
        pass

# --- 2. Concrete Strategy Implementations ---

class LevenshteinMetric(SimilarityMetric):
    """Calculates similarity based on Levenshtein (edit) distance."""
    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Levenshtein doesn't require pre-calculation, so we just return the original df.
        return df

    def calculate_scores(self, target_sequence: str, reference_df: pd.DataFrame) -> pd.Series:
        return reference_df['sequence'].apply(
            lambda ref_seq: Levenshtein.distance(target_sequence, ref_seq)
        )

    @property
    def sort_ascending(self) -> bool:
        # For distance, lower is better, so we sort ascending.
        return True