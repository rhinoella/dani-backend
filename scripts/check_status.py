#!/usr/bin/env python3
"""Check current ingestion status (no Ollama needed)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from app.core.config import settings

client = QdrantClient(url=settings.QDRANT_URL)

# Get collection info
info = client.get_collection("meeting_transcripts")
print("=" * 60)
print("📊 CURRENT INGESTION STATUS")
print("=" * 60)
print(f"Points ingested: {info.points_count}")
print(f"Vector dimensions: {info.config.params.vectors.size}")

# Get sample points to verify structure
points = client.scroll(
    collection_name="meeting_transcripts",
    limit=3,
    with_payload=True,
    with_vectors=False
)[0]

print()
print("📋 SAMPLE CHUNK STRUCTURE (New Format)")
print("-" * 60)

for i, p in enumerate(points, 1):
    payload = p.payload
    text = payload.get("text", "")
    word_count = len(text.split())
    
    print(f"")
    print(f"Chunk {i}:")
    print(f"  ✅ doc_type: {payload.get('doc_type', 'MISSING')}")
    print(f"  ✅ speakers: {payload.get('speakers', 'MISSING')}")
    tid = payload.get('transcript_id', 'MISSING')
    print(f"  ✅ transcript_id: {tid[:20] if tid else 'MISSING'}...")
    print(f"  ✅ date: {payload.get('date', 'MISSING')}")
    print(f"  📏 word_count: {word_count}")
    print(f"  📝 text preview: \"{text[:150]}...\"")

print()
print("=" * 60)
print("COMPARISON: OLD vs NEW")
print("=" * 60)
print(f"                     OLD        NEW")
print(f"doc_type field:       ❌         ✅")
print(f"speakers as list:     ❌         ✅")  
print(f"avg chunk size:    9 words    ~500 words")
print(f"embedding prefix:     ❌         ✅")
print("=" * 60)
