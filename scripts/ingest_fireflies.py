#!/usr/bin/env python3
"""
Ingest Fireflies transcripts into Qdrant for DANI Engine.
Optimized with: wave processing, deduplication, periodic upserts, parallel embeddings.

Uses nomic-embed-text with search_document: prefix for asymmetric search.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client.http import models as qm

from app.core.config import settings
from app.ingestion.loaders.fireflies_loader import FirefliesLoader
from app.ingestion.chunker import TokenChunker
from app.embeddings.client import OllamaEmbeddingClient
from app.vectorstore.qdrant import QdrantStore
from app.utils.id_generator import stable_point_id


# Configuration
WAVE_SIZE = 50  # Process this many transcripts per wave
UPSERT_BATCH = 100  # Upsert to Qdrant every N transcripts
EMBEDDING_BATCH_SIZE = 8  # Embed this many chunks in parallel (increased from 5)


async def embed_chunks_batch(texts: list[str], embedder: OllamaEmbeddingClient) -> list[list[float]]:
    """
    Embed all chunks in a single batch call with search_document prefix.
    This is critical for nomic-embed-text asymmetric search to work properly.
    """
    # Use embed_documents which adds the search_document: prefix automatically
    return await embedder.embed_documents(texts, batch_size=EMBEDDING_BATCH_SIZE)


async def main():
    print("🚀 Starting Fireflies ingestion (optimized)...")
    print("📝 Using nomic-embed-text with search_document: prefix for asymmetric search")

    loader = FirefliesLoader()
    embedder = OllamaEmbeddingClient()
    store = QdrantStore()
    # Use optimized chunk settings: 512 tokens, 150 overlap (Project Plan alignment)
    chunker = TokenChunker(chunk_size=512, overlap=150)

    # Ensure collection exists with correct vector dimensions
    vector_size = 768  # nomic-embed-text embedding size
    store.ensure_collection(settings.QDRANT_COLLECTION_TRANSCRIPTS, vector_size)

    skip = 0
    total_processed = 0
    total_skipped = 0
    total_ingested = 0
    points_buffer: list[qm.PointStruct] = []
    
    print(f"⚙️  Configuration:")
    print(f"   - Wave size: {WAVE_SIZE} transcripts")
    print(f"   - Upsert every: {UPSERT_BATCH} transcripts")
    print(f"   - Embedding batch size: {EMBEDDING_BATCH_SIZE} chunks")
    print(f"   - Chunk size: 350 tokens, overlap: 100 tokens\n")

    # 🌊 Process in waves: fetch 50 → process 50 → repeat
    wave_num = 1
    while True:
        print(f"🌊 Wave {wave_num}: Fetching {WAVE_SIZE} transcripts...")
        batch = await loader.list_transcripts(limit=WAVE_SIZE, skip=skip)
        
        if not batch:
            print("   No more transcripts to fetch")
            break
        
        print(f"   Fetched {len(batch)} transcripts")
        
        # Process each transcript in the wave
        for t in batch:
            total_processed += 1
            transcript_id = t["id"]
            source_file = f"fireflies:{transcript_id}"
            
            # ✅ Skip already-ingested transcripts
            if store.check_source_exists(settings.QDRANT_COLLECTION_TRANSCRIPTS, source_file):
                print(f"[{total_processed}] ⏭️  {t.get('title', transcript_id)[:50]} (already exists)")
                total_skipped += 1
                continue
            
            try:
                transcript = await loader.get_transcript(transcript_id)
                sentences = transcript.get("sentences", [])

                if not sentences:
                    print(f"[{total_processed}] ⚠️  {t.get('title', transcript_id)[:50]} (no sentences)")
                    total_skipped += 1
                    continue

                # Use speaker-aware chunking with structured sentences
                base_metadata = {
                    "transcript_id": transcript_id,
                    "title": transcript.get("title"),
                    "date": transcript.get("date"),
                    "organizer_email": transcript.get("organizer_email"),
                    "source_file": source_file,
                }

                chunked_records = chunker.chunk_with_speakers(sentences, base_metadata)
                texts = [c["text"] for c in chunked_records]

                # ⚡ Batch embedding (more efficient than parallel embed_one)
                vectors = await embed_chunks_batch(texts, embedder)

                # Build points
                for idx, (rec, vector) in enumerate(zip(chunked_records, vectors)):
                    payload = {
                        **rec["metadata"],
                        "text": rec["text"],
                        "chunk_index": idx,
                    }

                    # Use same ID generation as ingestion_service for consistency
                    section_id = rec["metadata"].get("chunk_index", idx)
                    point_id = stable_point_id("fireflies", transcript_id, str(section_id), str(idx))

                    points_buffer.append(
                        qm.PointStruct(
                            id=point_id,  # UUID hex string is valid for Qdrant
                            vector=vector,
                            payload=payload,
                        )
                    )

                print(f"[{total_processed}] ✅ {t.get('title', transcript_id)[:50]} ({len(chunked_records)} chunks)")
                total_ingested += 1
                
                # 💾 Periodic batch upsert
                if total_ingested % UPSERT_BATCH == 0 and points_buffer:
                    print(f"\n💾 Upserting {len(points_buffer)} points to Qdrant...")
                    store.upsert(settings.QDRANT_COLLECTION_TRANSCRIPTS, points_buffer)
                    points_buffer = []
                    print(f"   Progress: {total_ingested} ingested, {total_skipped} skipped\n")
                    
            except Exception as e:
                print(f"[{total_processed}] ❌ {t.get('title', transcript_id)[:50]}: {e}")
                total_skipped += 1
        
        skip += WAVE_SIZE
        wave_num += 1
        
        # Stop if we got less than requested (no more data)
        if len(batch) < WAVE_SIZE:
            break

    # Final upsert for remaining points
    if points_buffer:
        print(f"\n💾 Final upsert: {len(points_buffer)} points to Qdrant...")
        store.upsert(settings.QDRANT_COLLECTION_TRANSCRIPTS, points_buffer)

    print(f"\n✅ Ingestion complete!")
    print(f"   Total processed: {total_processed}")
    print(f"   Ingested: {total_ingested}")
    print(f"   Skipped: {total_skipped}")


if __name__ == "__main__":
    asyncio.run(main())
