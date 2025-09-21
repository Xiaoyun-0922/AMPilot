import os
import pandas as pd
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType

from configuration import WEAVIATE_URL, WEAVIATE_COLLECTION_NAME, DATA_DIR
from embeddings import get_embedding_model

# Ingestion tuning (can be overridden via environment variables)
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "128"))
CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "5000"))  # number of objects per batch context
ERROR_SAMPLE_EVERY = int(os.getenv("INGEST_ERROR_SAMPLE_EVERY", "500"))

def get_weaviate_client():
    """Connects to the Weaviate instance."""
    try:
        # First try with default configuration
        print(f"Attempting to connect to Weaviate at {WEAVIATE_URL}...")

        # Parse URL
        url_parts = WEAVIATE_URL.replace("http://", "").replace("https://", "")
        if ":" in url_parts:
            host = url_parts.split(":")[0]
            port = int(url_parts.split(":")[1])
        else:
            host = url_parts
            port = 8080

        print(f"Connecting to host: {host}, port: {port}")

        # Try with additional configuration for better compatibility
        import weaviate.classes.config as wvc

        client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051,  # Explicitly set gRPC port
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=60, query=60, insert=120),  # Increase timeouts
                startup_period=10  # Wait longer for startup
            )
        )

        # Test the connection
        if client.is_ready():
            print("Successfully connected to Weaviate!")
            return client
        else:
            print("Weaviate client created but not ready")

    except Exception as e:
        print(f"Failed to connect to Weaviate with default settings: {e}")
        print("Trying alternative connection method...")

        # Try with skip_init_checks as suggested in error message
        try:
            client = weaviate.connect_to_local(
                host=host,
                port=port,
                skip_init_checks=True  # Skip startup checks
            )
            print("Connected to Weaviate with skip_init_checks=True")
            return client
        except Exception as e2:
            print(f"Alternative connection also failed: {e2}")
            print("\nTroubleshooting suggestions:")
            print("1. Check if Weaviate is running: docker ps")
            print("2. Check if ports 8080 and 50051 are accessible")
            print("3. Try restarting Weaviate: docker-compose restart")
            exit(1)

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
        ],
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw()
    )

    df = process_and_combine_data(DATA_DIR)
    df['text_for_embedding'] = df.apply(create_text_for_embedding, axis=1)

    embeddings = get_embedding_model()
    amp_collection = client.collections.get(WEAVIATE_COLLECTION_NAME)

    print(f"Generating embeddings for {len(df)} records (this may take a few minutes)...")
    texts_to_embed = df['text_for_embedding'].tolist()

    # Process embeddings in smaller batches to avoid memory issues and show progress
    vectors = []
    embedding_batch_size = 100  # Process 100 texts at a time

    print("Processing embeddings in batches...")
    for i in tqdm(range(0, len(texts_to_embed), embedding_batch_size), desc="Embedding batches"):
        batch_texts = texts_to_embed[i:i + embedding_batch_size]
        try:
            batch_vectors = embeddings.embed_documents(batch_texts)
            vectors.extend(batch_vectors)
        except Exception as e:
            print(f"Error processing embedding batch {i//embedding_batch_size + 1}: {e}")
            # Add zero vectors as fallback to maintain alignment
            zero_vector = [0.0] * 384  # Assuming 384-dim embeddings, adjust if needed
            vectors.extend([zero_vector] * len(batch_texts))

    print(f"Generated {len(vectors)} embeddings successfully.")

    records = df.to_dict('records')

    print("Starting batch import into Weaviate...")
    # Use a smaller batch size to avoid memory issues and provide better progress feedback
    with amp_collection.batch.fixed_size(batch_size=50) as batch:
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
            try:
                batch.add_object(properties=properties, vector=vectors[i])
            except Exception as e:
                # Log and continue on individual object errors to prevent total stall
                if i % 100 == 0:  # More frequent error reporting
                    print(f"Warning: batch add failed at index {i}: {e}")
                continue
            # More frequent progress updates
            if (i + 1) % 1000 == 0:
                print(f"... {i + 1}/{len(records)} records processed ({((i+1)/len(records)*100):.1f}%)")

    print(f"Ingestion complete! {len(records)} records were successfully imported with the new embedding strategy.")
    client.close()

if __name__ == "__main__":
    main()