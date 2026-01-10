#!/usr/bin/env python3
"""
Initialize Qdrant collections for DANI Engine.
Run this script to set up the vector database collections.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.embeddings.client import OllamaEmbeddingClient
from app.vectorstore.qdrant import QdrantStore


async def main():
    print("🚀 Initializing Qdrant collections...")
    
    store = QdrantStore()
    embedder = OllamaEmbeddingClient()
    
    # Get embedding dimension
    print("📏 Getting embedding dimension from Ollama...")
    test_vec = await embedder.embed_one("test")
    vector_size = len(test_vec)
    print(f"   Vector size: {vector_size}")
    
    # Create collections
    collections = [
        settings.QDRANT_COLLECTION_TRANSCRIPTS,
        settings.QDRANT_COLLECTION_DOCUMENTS,
    ]
    
    for collection in collections:
        print(f"📦 Creating collection: {collection}")
        try:
            store.ensure_collection(name=collection, vector_size=vector_size)
            print(f"   ✅ Collection '{collection}' is ready")
        except Exception as e:
            print(f"   ❌ Error creating '{collection}': {e}")
            return 1
    
    print("\n✨ All collections initialized successfully!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
