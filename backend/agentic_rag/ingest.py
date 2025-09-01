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
    """Reads all CSV (.csv) files from a directory and combines them."""
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"No CSV (.csv) files found in the '{data_dir}' directory.")
        exit()
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat([df for df in df_list if not df.empty], ignore_index=True).fillna("N/A")
    print(f"Loaded a total of {len(combined_df)} AMP records from CSV files.")
    return combined_df

def create_text_for_embedding(row):
    """
    Formats a row of data into a descriptive text block for embedding,
    using only the most relevant semantic fields.
    """
    text_parts = [
        f"ID: {row.get('DRAMP_ID', 'N/A')}",
        f"Name: {row.get('Name', 'N/A')}",
        f"Description: {row.get('Description', 'N/A')}",
    ]
    return "\n\n".join(part for part in text_parts if "N/A" not in part)

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
            Property(name="dramp_id", data_type=DataType.TEXT),
            Property(name="sequence", data_type=DataType.TEXT),
            Property(name="name", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="reference", data_type=DataType.TEXT),
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
                'dramp_id': record.get('DRAMP_ID', 'N/A'),
                'sequence': record.get('Sequence', 'N/A'),
                'name': record.get('Name', 'N/A'),
                'description': record.get('Description', 'N/A'),
                'reference': record.get('Reference', 'N/A'),
            }
            batch.add_object(properties=properties, vector=vectors[i])
            
    print(f"Ingestion complete! {len(records)} records were successfully imported with the new embedding strategy.")
    client.close()

if __name__ == "__main__":
    main()