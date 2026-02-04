
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.core.config import settings
from app.embeddings.factory import get_embedding_client
from app.vectorstore.qdrant import QdrantStore

async def main():
    print(f"Checking Qdrant URL: {settings.QDRANT_URL}")
    store = QdrantStore()
    embedder = get_embedding_client()
    
    query_text = "somatosensory"
    print(f"\nSearching for '{query_text}' in 'documents' collection...")
    
    try:
        # Generate embedding
        print("Generating embedding...")
        query_vector = await embedder.embed_query(query_text)
        
        # Search
        results = store.client.query_points(
            collection_name=settings.QDRANT_COLLECTION_DOCUMENTS,
            query=query_vector,
            limit=5,
            with_payload=True
        ).points
        
        print(f"\nFound {len(results)} results:")
        for hit in results:
            payload = hit.payload or {}
            text = payload.get("text", "")
            title = payload.get("title") or payload.get("filename")
            print(f"\n[Score: {hit.score:.4f}] {title}")
            print(f"Excerpt: {text[:200]}...")
            
    except Exception as e:
        print(f"Error searching: {e}")

if __name__ == "__main__":
    asyncio.run(main())
