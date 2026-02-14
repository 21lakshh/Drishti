import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "building_knowledge")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def create_index():
    print(f"--- Creating Index for Collection: {COLLECTION_NAME} ---")
    
    # We use the synchronous client for setup tasks
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # This command creates the index. It doesn't change your data, 
    # it just makes 'user_id' searchable/filterable.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    print(f"✅ Index created for field 'user_id' in '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    create_index()