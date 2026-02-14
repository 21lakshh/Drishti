import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

# Configuration
PDF_PATH = "data/building_guide.pdf"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "drishti")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hardcoded User ID for Multitenancy
USER_ID = 1

def ingest_data():
    print(f"--- Starting Ingestion for User ID: {USER_ID} ---")
    
    # 1. Initialize Clients
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # 2. Check/Create Collection
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,  # text-embedding-3-small dimension
                distance=models.Distance.COSINE
            )
        )

    # 3. Load and Chunk PDF
    if not os.path.exists(PDF_PATH):
        print(f"Error: File {PDF_PATH} not found.")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Generated {len(splits)} chunks.")

    # 4. Generate Embeddings & Prepare Points
    points = []
    print("Generating embeddings and preparing vectors...")
    
    for split in splits:
        # Generate Embedding
        response = openai_client.embeddings.create(
            input=split.page_content,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        # Create Point
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),  # Generate unique ID
            vector=embedding,
            payload={
                "text": split.page_content,
                "source": PDF_PATH,
                "page": split.metadata.get("page", 0),
                "user_id": USER_ID # Metadata for Multitenancy filtering
            }
        ))

    # 5. Upsert to Qdrant
    print(f"Upserting {len(points)} points to Qdrant...")
    operation_info = qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Ingestion Complete. Status: {operation_info.status}")

if __name__ == "__main__":
    ingest_data()