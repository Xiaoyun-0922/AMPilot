"""
Handles data loading, processing, and retrieval from the AMP database.
"""
import random
import pandas as pd
from typing import List, Dict, Optional

import config
from metrics import SimilarityMetric, LevenshteinMetric

# --- 3. Refactored DataHandler ---

class DataHandler:
    """
    A class to manage the AMP database stored in a CSV file.
    """
    def __init__(
            self, 
            db_path: str = config.DATABASE_PATH,
            metric: Optional[SimilarityMetric] = None
    ):
        """
        Initializes the DataHandler by loading the database.

        Args:
            db_path (str): Path to the CSV database file.
            metric (SimilarityMetric, optional): The similarity metric strategy to use.
                                                  Defaults to LevenshteinMetric.
        """
        self.metric = metric if metric is not None else LevenshteinMetric()
        try:
            df = pd.read_csv(db_path)

            # IMPORTANT: Prepare the dataframe using the chosen metric.
            # This pre-calculates embeddings, descriptors, etc.
            self.df = self.metric.prepare_dataframe(df)

            # Pre-filter the dataframe to separate positive (AMP) and negative (non-AMP) samples.
            self.amp_df = self.df[self.df['is_amp'] == True].copy()
            self.non_amp_df = self.df[self.df['is_amp'] == False].copy()

            # No need to prepare amp_df and non_amp_df again as self.df already has the data.
            # self.amp_df = self.metric.prepare_dataframe(self.amp_df)
            # self.non_amp_df = self.metric.prepare_dataframe(self.non_amp_df)

            print(f"Successfully loaded database from {db_path} with {len(self.df)} total entries.")
            print(f"Found {len(self.amp_df)} AMPs (is_amp=True) to be used as references.")
            print(f"Found {len(self.non_amp_df)} non-AMPs (is_amp=False) to be used as references.")
        except FileNotFoundError:
            print(f"Error: Database file not found at {db_path}.")
            print("Please ensure the CSV file exists and the path in config.py is correct.")
            self.df = pd.DataFrame() # Create an empty dataframe to avoid errors
            self.amp_df = pd.DataFrame()
            self.non_amp_df = pd.DataFrame()

    def find_references(self, 
                        target_sequence: str, 
                        num_references: int = 4, 
                        mode: str = 'balanced_closest'
                        ) -> List[Dict]:
        """
        Finds a set of references from the database based on a specified mode.

        Modes:
        - 'balanced_closest': Finds a balanced set of the most similar positive refs (using the chosen metric) and random negative refs.
        - 'random': Finds a random sample of both positive and negative sequences.

        Args:
            target_sequence (str): The sequence of the target AMP.
            num_references (int): The total number of references to return (should be an even number).
            mode (str): The selection mode.

        Returns:
            A list of dictionaries, where each dictionary represents a reference AMP.
        """
        if self.amp_df.empty or self.non_amp_df.empty:
            print("Warning: AMP or non-AMP dataframe is empty. Cannot find references.")
            return []

        num_pos_refs = num_references // 2
        num_neg_refs = num_references - num_pos_refs

        # --- Find Positive References (AMPs) ---
        positive_refs = []
        # Ensure the target sequence itself is not selected as a reference
        amp_pool = self.amp_df[self.amp_df['sequence'] != target_sequence].copy()
        if len(amp_pool) >= num_pos_refs:
            if 'random' in mode:
                positive_refs = amp_pool.sample(n=num_pos_refs).to_dict('records')
            elif 'closest' in mode:
                # Create a copy here to safely add the 'score' column
                amp_pool_copy = amp_pool.copy()
                # Use the injected metric strategy to calculate scores
                amp_pool_copy['score'] = self.metric.calculate_scores(target_sequence, amp_pool_copy)
                # Sort by distance and take the top N
                closest_pos = amp_pool_copy.sort_values(by='score', ascending=self.metric.sort_ascending).head(num_pos_refs)
                positive_refs = closest_pos.drop(columns=['score']).to_dict('records')
            else:
                raise ValueError(f"Unknown mode: '{mode}'.")

        # --- Find Negative References (non-AMPs) ---
        negative_refs = []
        non_amp_pool = self.non_amp_df[self.non_amp_df['sequence'] != target_sequence]
        if len(non_amp_pool) >= num_neg_refs:
            # For negative samples, random selection is usually sufficient
            negative_refs = non_amp_pool.sample(n=num_neg_refs).to_dict('records')
        
        # Combine and return
        all_references = positive_refs + negative_refs
        # Shuffle the final list so positive and negative samples are mixed
        random.shuffle(all_references)
        return all_references

    def get_ground_truth(self, sequence: str) -> Optional[Dict]:
        """
        Retrieves the ground truth data (MIC, HC50) for a given sequence.

        Args:
            sequence (str): The AMP sequence to look up.

        Returns:
            A dictionary with the ground truth data or None if not found.
        """
        if self.df.empty:
            return None

        result = self.df[self.df['sequence'] == sequence]
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

if __name__ == '__main__':
    # --- Simple Test ---
    print("\n--- Running a simple test for DataHandler ---")
    
    # --- Test 1: Levenshtein Metric (default) ---
    print("\n--- Initializing with LevenshteinMetric ---")
    lev_handler = DataHandler(metric=LevenshteinMetric())
    # Let's pick one from the database to see if 'closest' works well
    target_seq = "FLPLLLGKVVCAITKKC" 
    print(f"\nTarget Sequence: {target_seq}")
    print("\n--- Testing 'balanced_closest' mode with Levenshtein ---")
    closest_refs = lev_handler.find_references(target_seq, num_references=4, mode='balanced_closest')
    if closest_refs:
        for i, ref in enumerate(closest_refs):
            print(f"Reference {i+1}:")
            print(f"  Sequence: {ref.get('sequence')}")
            print(f"  Is AMP: {ref.get('is_amp')}")
    else:
        print("Could not find references.")

