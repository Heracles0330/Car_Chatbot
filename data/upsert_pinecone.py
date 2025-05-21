import json
import os
from pinecone import Pinecone, ServerlessSpec # PodSpec can be imported if needed
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# !!! IMPORTANT: Fill these in with your details !!!
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # e.g., "gcp-starter", "us-west1-gcp"
PINECONE_INDEX_NAME = "products"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Your OpenAI API Key

# OpenAI's text-embedding-ada-002 model has a dimension of 1536
# This MUST match the dimension of your Pinecone index.
EXPECTED_DIMENSION = 1536
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

PRODUCTS_JSON_PATH = os.path.join("data", "row_data", "products.json")

# Batch size for upserting to Pinecone and processing OpenAI embeddings
BATCH_SIZE = 100 # Pinecone's recommended max batch size for upserts.
# OpenAI API has rate limits, so processing in smaller text batches might be needed if you hit them.
# For embedding, we'll process texts individually for simplicity here, but batching for OpenAI API is also possible.

def initialize_pinecone(api_key):
    """Initializes and returns a Pinecone client instance."""
    if not api_key or api_key == "YOUR_PINECONE_API_KEY":
        raise ValueError("Pinecone API key is not set. Please update PINECONE_API_KEY.")
    
    pc = Pinecone(api_key=api_key)
    return pc

def get_or_create_pinecone_index(pc_instance, index_name, dimension):
    """Gets a Pinecone index. Warns if it doesn't exist or dimension mismatches."""
    if not index_name or index_name == "YOUR_PINECONE_INDEX_NAME":
        raise ValueError("Pinecone index name is not set. Please update PINECONE_INDEX_NAME.")

    pc_instance.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    index = pc_instance.Index(index_name)
    try:
        index_stats = index.describe_index_stats()
        # Pinecone describe_index_stats() might not directly return dimension for all index types easily.
        # More reliably, check index description.
        index_description = pc_instance.describe_index(index_name)
        actual_dimension = index_description.dimension

        if actual_dimension != dimension:
            raise ValueError(
                f"Pinecone index '{index_name}' has dimension {actual_dimension}, "
                f"but the OpenAI model '{OPENAI_EMBEDDING_MODEL}' produces embeddings of dimension {dimension}. "
                f"Please ensure the index dimension matches the model."
            )
    except Exception as e:
        print(f"Could not fully verify index dimension for '{index_name}'. Please ensure it is {dimension}. Error: {e}")
        # Proceed with caution if dimension check fails.
        
    return index

def get_openai_embedding(text, client, model="text-embedding-ada-002"):
    """Generates an embedding for the given text using OpenAI API."""
    try:
        text = text.replace("\n", " ") # API best practice
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding from OpenAI for text snippet '{text[:50]}...': {e}")
        # Consider adding retries or more sophisticated error handling here
        return None

def load_products(json_path):
    """Loads products from the specified JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "products" in data and isinstance(data["products"], list):
            return data["products"]
        else:
            print(f"Warning: Could not find a 'products' list in {json_path}. Assuming root is a list.")
            if isinstance(data, list): return data
            raise ValueError("Data in JSON is not a list of products.")
    except FileNotFoundError:
        print(f"Error: Products JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return []

def main():
    print("Starting Pinecone upsert process using OpenAI embeddings...")

    # 0. Initialize OpenAI client
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("ERROR: OpenAI API Key is not set. Please update OPENAI_API_KEY in the script.")
        return
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # 1. Initialize Pinecone
    try:
        pc = initialize_pinecone(PINECONE_API_KEY)
        index = get_or_create_pinecone_index(pc, PINECONE_INDEX_NAME, EXPECTED_DIMENSION)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    except Exception as e:
        print(f"Error initializing Pinecone or getting index: {e}")
        return

    # 2. Load Products
    products = load_products(PRODUCTS_JSON_PATH)
    if not products:
        print("No products to process.")
        return
    print(f"Loaded {len(products)} products from {PRODUCTS_JSON_PATH}.")

    # 3. Prepare and Upsert Data in Batches
    print(f"Generating embeddings with OpenAI ('{OPENAI_EMBEDDING_MODEL}') and upserting to Pinecone...")
    vectors_to_upsert = []
    processed_count = 0
    successful_upserts = 0

    for i, product in enumerate(products):
        if not isinstance(product, dict):
            print(f"Skipping item at index {i} as it's not a dictionary: {product}")
            continue

        product_id = product.get("id")
        title = product.get("title", "")
        description = product.get("description", "")

        if not product_id:
            print(f"Skipping product due to missing 'id': {product.get('sku', 'Unknown SKU')}")
            continue

        # Combine title and description for embedding
        text_to_embed = f"Title: {title}\nDescription: {description}".strip()
        if not text_to_embed or text_to_embed == "Title: \nDescription:":
            print(f"Skipping product id '{product_id}' due to empty title and description.")
            continue
        
        # Introduce a small delay to help with potential rate limits if processing many items quickly
        if i > 0 and i % 10 == 0: # Every 10 items
            time.sleep(0.5) # Sleep for 0.5 seconds

        embedding = get_openai_embedding(text_to_embed, openai_client, model=OPENAI_EMBEDDING_MODEL)

        if embedding is None:
            print(f"Failed to get embedding for product id '{product_id}'. Skipping.")
            continue
            
        # Prepare metadata: include all fields from the product
        metadata = {k: v for k, v in product.items() if v is not None}

        vectors_to_upsert.append({
            "id": str(product_id), 
            "values": embedding,
            "metadata": metadata
        })
        processed_count +=1

        if len(vectors_to_upsert) >= BATCH_SIZE:
            try:
                index.upsert(vectors=vectors_to_upsert)
                successful_upserts += len(vectors_to_upsert)
                print(f"Upserted batch of {len(vectors_to_upsert)}. Total successful upserts: {successful_upserts}")
                vectors_to_upsert = []
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {e}")
                # Consider saving failed batches for retry
                
    # Upsert any remaining vectors
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            successful_upserts += len(vectors_to_upsert)
            print(f"Upserted final batch of {len(vectors_to_upsert)}. Total successful upserts: {successful_upserts}")
        except Exception as e:
            print(f"Error upserting final batch to Pinecone: {e}")

    print(f"\nPinecone upsert process finished.")
    print(f"Total products considered for embedding: {processed_count}/{len(products)}")
    print(f"Total vectors successfully upserted to Pinecone: {successful_upserts}")
    
    try:
        print("\nFinal index stats:")
        print(index.describe_index_stats())
    except Exception as e:
        print(f"Could not fetch final index stats: {e}")

if __name__ == "__main__":
    print("Before running this script, ensure you have the following libraries installed:")
    print("  pip install pinecone-client openai") # Removed sentence-transformers, added openai
    print("Also, make sure you have filled in your Pinecone API Key, Environment, Index Name, and OpenAI API Key in the script.\n")
    
    if "YOUR_PINECONE_API_KEY" in PINECONE_API_KEY or \
       "YOUR_PINECONE_INDEX_NAME" in PINECONE_INDEX_NAME or \
       "YOUR_OPENAI_API_KEY" in OPENAI_API_KEY:
        print("ERROR: Please fill in your Pinecone and OpenAI API Keys, Pinecone Environment, and Index Name in the script before running.")
    else:
        main()
