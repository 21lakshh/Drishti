import asyncio
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from openai import AsyncOpenAI

load_dotenv()

# Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "building_knowledge")
USER_ID = 1

async def test_retrieval():
    print("--- Starting RAG Retrieval Test ---")
    
    # 1. Initialize Clients
    qdrant_client = AsyncQdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 2. Define Test Query
    object_to_find = "shoes"
    user_location = "bed"
    query_text = f"Where is {object_to_find}? I am currently at {user_location}."
    
    print(f"Query: '{query_text}'")

    try:
        # 3. Generate Embedding
        print("Generating embedding...")
        embedding_response = await openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_vector = embedding_response.data[0].embedding

        # 4. Query Qdrant (Using query_points instead of search)
        print(f"Searching collection '{COLLECTION_NAME}' for User ID {USER_ID}...")
        
        # Use query_points (Universal Query API)
        results = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,  # Pass vector here
            limit=3,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(
                            value=USER_ID,
                        ),
                    )
                ]
            ),
        )

        # 5. Display Results
        # Note: query_points returns a QueryResponse object containing 'points'
        points = results.points 

        print(f"\nFound {len(points)} results:\n")
        for i, hit in enumerate(points):
            score = hit.score
            text = hit.payload.get("text", "No text found")
            source = hit.payload.get("source", "Unknown source")
            print(f"--- Result {i+1} (Score: {score:.4f}) ---")
            print(f"Source: {source}")
            print(f"Content: {text[:200]}...") 
            print("-" * 30)

        if not points:
            print("❌ No results found. Check ingestion or filters.")

    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await qdrant_client.close()

if __name__ == "__main__":
    asyncio.run(test_retrieval())