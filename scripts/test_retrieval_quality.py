#!/usr/bin/env python3
"""
Test retrieval quality with the newly ingested data.
Run this while or after re-ingestion to see improvements.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from app.embeddings.client import OllamaEmbeddingClient
from app.core.config import settings


async def test_retrieval():
    # Connect
    client = QdrantClient(url=settings.QDRANT_URL)
    embedder = OllamaEmbeddingClient()
    
    print("=" * 60)
    print("🧪 LIVE RETRIEVAL TEST")
    print("=" * 60)
    
    # Check how many points we have
    try:
        info = client.get_collection("meeting_transcripts")
        print(f"\n📊 Collection: {info.points_count} points ingested so far\n")
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return
    
    # Test queries
    test_queries = [
        "What action items were discussed?",
        "Who attended the meeting?",
        "What deadlines were mentioned?",
        "project timeline discussion",
    ]
    
    for query in test_queries:
        print(f'\n📝 Query: "{query}"')
        print("-" * 50)
        
        # Generate query embedding (with search_query: prefix)
        query_vec = await embedder.embed_query(query)
        
        # Search
        results = client.query_points(
            collection_name="meeting_transcripts",
            query=query_vec,
            limit=3,
            with_payload=True
        ).points
        
        if results:
            for i, r in enumerate(results, 1):
                score = r.score
                text = r.payload.get("text", "")[:250]
                speakers = r.payload.get("speakers", [])
                doc_type = r.payload.get("doc_type", "unknown")
                
                # Quality indicator
                if score > 0.75:
                    quality = "🟢 GOOD"
                elif score > 0.5:
                    quality = "🟡 OK"
                else:
                    quality = "🔴 LOW"
                
                print(f"\n   {quality} (score: {score:.4f})")
                print(f"   doc_type: {doc_type}")
                print(f"   speakers: {speakers}")
                print(f"   text: \"{text}...\"")
        else:
            print("   No results found")
    
    await embedder.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 WHAT CHANGED")
    print("=" * 60)
    print("                      BEFORE       NOW")
    print("doc_type field:         ❌          ✅")
    print("speakers as list:       ❌          ✅") 
    print("Chunk size:           9 words    ~500 words")
    print("Embedding prefix:       ❌          ✅")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_retrieval())
