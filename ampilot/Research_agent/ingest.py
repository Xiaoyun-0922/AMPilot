# backend/ingest.py

import os
import pandas as pd
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType

from configuration import WEAVIATE_URL, WEAVIATE_COLLECTION_NAME, DATA_DIR
from embeddings import get_embedding_model

def get_weaviate_client():
    """Connects to the Weaviate instance."""
    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_URL.replace("http://", "").split(":")[0],
            port=int(WEAVIATE_URL.replace("http://", "").split(":")[1])
        )
        print("Successfully connected to Weaviate!")
        return client
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        exit()

def process_and_combine_data(data_dir: str) -> pd.DataFrame:
    """Reads GRAMPA CSV file and processes it for Weaviate ingestion."""
    grampa_file = os.path.join(data_dir, "grampa.csv")

    if not os.path.exists(grampa_file):
        print(f"GRAMPA file not found: {grampa_file}")
        print("Please ensure grampa.csv is in the data directory.")
        exit()

    print(f"Loading GRAMPA data from {grampa_file}...")
    df = pd.read_csv(grampa_file)

    # Clean and process the data
    # Ensure numeric type for MIC log values
    df['value'] = pd.to_numeric(df.get('value'), errors='coerce')

    # Convert MIC value from log10 scale to actual concentration (µM); keep None where unknown
    df['mic_value_um'] = (10 ** df['value']).where(df['value'].notna(), None)

    # Create readable modification description
    def format_modifications(row):
        def is_true(v):
            s = str(v).strip().lower()
            return s in ("true", "1", "yes", "y", "t")
        mods_cell = row.get('modifications')
        if pd.isna(mods_cell) or mods_cell in ['[]', 'N/A']:
            base = "No modifications"
        else:
            base = f"Specific: {mods_cell}"
        mods = []
        if is_true(row.get('has_cterminal_amidation')):
            mods.append("C-terminal amidation")
        if is_true(row.get('has_unusual_modification')):
            mods.append("Unusual modifications")
        extra = "; ".join(mods)
        if extra and base != "No modifications":
            return f"{base}; {extra}"
        return extra or base

    df['modification_description'] = df.apply(format_modifications, axis=1)

    print(f"Loaded a total of {len(df)} AMP-bacterium interaction records from GRAMPA.")
    return df

def create_text_for_embedding(row):
    """
    Formats a GRAMPA row into a descriptive text block for embedding,
    focusing on peptide sequence, target bacterium, and antimicrobial properties.
    """
    text_parts = [
        f"Peptide sequence: {row.get('sequence', 'N/A')}",
        f"Target bacterium: {row.get('bacterium', 'N/A')}",
        f"Bacterial strain: {row.get('strain', 'N/A')}",
        f"MIC concentration: {row.get('mic_value_um', 'N/A')} µM",
        f"Peptide modifications: {row.get('modification_description', 'N/A')}",
        f"Database source: {row.get('database', 'N/A')}"
    ]
    return " | ".join(part for part in text_parts if "N/A" not in part)

def _parse_bool(val) -> bool:
    """Robustly parse boolean-like values from CSV (True/False/1/0/'true'/'false'/'N/A')."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return val != 0
    s = str(val).strip().lower()
    if s in ("true", "yes", "y", "1", "t"): return True
    if s in ("false", "no", "n", "0", "f", "n/a", "", "none"): return False
    return False

def _safe_text(val):
    """Return cleaned text or None if missing/invalid."""
    if val is None:
        return None
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass
    s = str(val).strip()
    if s.lower() in ("n/a", "none", "nan", ""):
        return None
    return s

def _safe_float(val):
    """Return float or None if missing/invalid."""
    try:
        if val is None:
            return None
        import math
        if isinstance(val, float) and math.isnan(val):
            return None
        return float(val)
    except Exception:
        return None

def main():
    """Main ingestion script."""
    client = get_weaviate_client()
    
    if client.collections.exists(WEAVIATE_COLLECTION_NAME):
        print(f"Collection '{WEAVIATE_COLLECTION_NAME}' already exists. Deleting it for fresh import.")
        client.collections.delete(WEAVIATE_COLLECTION_NAME)
    
    print(f"Creating collection '{WEAVIATE_COLLECTION_NAME}' with the new schema...")
    
    client.collections.create(
        name=WEAVIATE_COLLECTION_NAME,
        properties=[
            Property(name="sequence", data_type=DataType.TEXT),
            Property(name="bacterium", data_type=DataType.TEXT),
            Property(name="strain", data_type=DataType.TEXT),
            Property(name="mic_value_um", data_type=DataType.NUMBER),
            Property(name="mic_log_value", data_type=DataType.NUMBER),
            Property(name="modifications", data_type=DataType.TEXT),
            Property(name="modification_description", data_type=DataType.TEXT),
            Property(name="is_modified", data_type=DataType.BOOL),
            Property(name="has_cterminal_amidation", data_type=DataType.BOOL),
            Property(name="has_unusual_modification", data_type=DataType.BOOL),
            Property(name="database", data_type=DataType.TEXT),
            Property(name="url_source", data_type=DataType.TEXT),
        ]
    )
    
    df = process_and_combine_data(DATA_DIR)
    df['text_for_embedding'] = df.apply(create_text_for_embedding, axis=1)

    embeddings = get_embedding_model()
    amp_collection = client.collections.get(WEAVIATE_COLLECTION_NAME)
    
    print(f"Generating embeddings for {len(df)} records (this may take a few minutes on GPU)...")
    texts_to_embed = df['text_for_embedding'].tolist()
    
    # <<< FIX: Removed the unsupported 'batch_size' keyword argument >>>
    # The underlying sentence-transformers library handles batching automatically and efficiently.
    vectors = embeddings.embed_documents(texts_to_embed)
    
    records = df.to_dict('records')

    print("Starting batch import into Weaviate...")
    with amp_collection.batch.dynamic() as batch:
        for i, record in enumerate(tqdm(records, desc="Importing progress")):
            properties = {
                'sequence': _safe_text(record.get('sequence')),
                'bacterium': _safe_text(record.get('bacterium')),
                'strain': _safe_text(record.get('strain')),
                'mic_value_um': _safe_float(record.get('mic_value_um')),
                'mic_log_value': _safe_float(record.get('value')),
                'modifications': _safe_text(record.get('modifications')),
                'modification_description': _safe_text(record.get('modification_description')),
                'is_modified': _parse_bool(record.get('is_modified', False)),
                'has_cterminal_amidation': _parse_bool(record.get('has_cterminal_amidation', False)),
                'has_unusual_modification': _parse_bool(record.get('has_unusual_modification', False)),
                'database': _safe_text(record.get('database')),
                'url_source': _safe_text(record.get('url_source')),
            }
            batch.add_object(properties=properties, vector=vectors[i])
            
    print(f"Ingestion complete! {len(records)} records were successfully imported with the new embedding strategy.")
    client.close()

if __name__ == "__main__":
    main()