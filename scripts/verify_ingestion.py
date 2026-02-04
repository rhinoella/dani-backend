#!/usr/bin/env python3
"""
Verification script to validate ingestion quality after RAG fixes.
Run this AFTER re-ingesting data to confirm everything is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from app.core.config import settings
from app.embeddings.factory import get_embedding_client


def check_mark(passed: bool) -> str:
    return "✅" if passed else "❌"


async def verify_ingestion():
    print("=" * 60)
    print("🔍 INGESTION VERIFICATION REPORT")
    print("=" * 60)
    
    # Parse host and port from URL
    qdrant_url = settings.QDRANT_URL
    if "://" in qdrant_url:
        qdrant_url = qdrant_url.split("://")[1]
    host, port = qdrant_url.split(":") if ":" in qdrant_url else (qdrant_url, 6333)
    
    client = QdrantClient(host=host, port=int(port))
    embedder = get_embedding_client()
    
    all_passed = True
    
    # ─────────────────────────────────────────────────────────────
    # 1. CHECK COLLECTION EXISTS
    # ─────────────────────────────────────────────────────────────
    print("\n📦 1. COLLECTION STATUS")
    print("-" * 40)
    
    try:
        info = client.get_collection("meeting_transcripts")
        point_count = info.points_count
        print(f"   Collection: meeting_transcripts")
        print(f"   Points: {point_count}")
        print(f"   Vector Dim: {info.config.params.vectors.size}")
        
        if point_count == 0:
            print(f"   {check_mark(False)} No points found - run reingest_all.py first!")
            all_passed = False
        else:
            print(f"   {check_mark(True)} Collection has data")
    except Exception as e:
        print(f"   {check_mark(False)} Collection error: {e}")
        all_passed = False
        return
    
    # ─────────────────────────────────────────────────────────────
    # 2. CHECK PAYLOAD STRUCTURE
    # ─────────────────────────────────────────────────────────────
    print("\n📋 2. PAYLOAD STRUCTURE")
    print("-" * 40)
    
    # Get a sample point
    points = client.scroll(
        collection_name="meeting_transcripts",
        limit=5,
        with_payload=True,
        with_vectors=False
    )[0]
    
    if not points:
        print(f"   {check_mark(False)} No points to check")
        all_passed = False
    else:
        sample = points[0].payload
        
        # Check required fields
        required_fields = ["transcript_id", "text", "chunk_index", "meeting_title", "date"]
        for field in required_fields:
            has_field = field in sample
            print(f"   {check_mark(has_field)} Has '{field}' field")
            if not has_field:
                all_passed = False
        
        # Check NEW fields from our fixes
        print("\n   🆕 New fields from fixes:")
        
        # doc_type field
        has_doc_type = "doc_type" in sample
        print(f"   {check_mark(has_doc_type)} Has 'doc_type' field", end="")
        if has_doc_type:
            print(f" (value: {sample['doc_type']})")
        else:
            print(" - MISSING! Re-ingest required")
            all_passed = False
        
        # speakers field (list, not string)
        has_speakers = "speakers" in sample
        speakers_is_list = has_speakers and isinstance(sample.get("speakers"), list)
        print(f"   {check_mark(speakers_is_list)} Has 'speakers' as list", end="")
        if has_speakers:
            print(f" (value: {sample['speakers']})")
        else:
            print(" - MISSING! Re-ingest required")
            all_passed = False
        
        # Check for OLD field that should NOT exist
        has_old_speaker = "speaker" in sample
        if has_old_speaker:
            print(f"   {check_mark(False)} OLD 'speaker' field still present - Re-ingest required!")
            all_passed = False
    
    # ─────────────────────────────────────────────────────────────
    # 3. CHECK EMBEDDING PREFIX (Critical Fix)
    # ─────────────────────────────────────────────────────────────
    print("\n🔤 3. EMBEDDING PREFIX VERIFICATION")
    print("-" * 40)
    
    # Get a point with its vector
    points_with_vectors = client.scroll(
        collection_name="meeting_transcripts",
        limit=1,
        with_payload=True,
        with_vectors=True
    )[0]
    
    if points_with_vectors:
        sample_point = points_with_vectors[0]
        sample_text = sample_point.payload.get("text", "")[:100]
        stored_vector = sample_point.vector
        
        # Generate embedding WITH prefix (correct way)
        query_with_prefix = await embedder.embed_query(sample_text)
        
        # Generate embedding WITHOUT prefix (old broken way) - simulate raw embedding
        # by embedding text directly without using the query method
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as raw_client:
            resp = await raw_client.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": settings.EMBEDDING_MODEL, "prompt": sample_text}
            )
            query_without_prefix = resp.json().get("embedding") if resp.status_code == 200 else None
        
        # Calculate similarities
        from numpy import dot
        from numpy.linalg import norm
        
        def cosine_sim(a, b):
            return dot(a, b) / (norm(a) * norm(b))
        
        sim_with_prefix = cosine_sim(stored_vector, query_with_prefix)
        
        print(f"   Sample text: '{sample_text[:50]}...'")
        print(f"   Similarity with search_query: prefix: {sim_with_prefix:.4f}")
        
        if query_without_prefix:
            sim_without_prefix = cosine_sim(stored_vector, query_without_prefix)
            print(f"   Similarity without prefix: {sim_without_prefix:.4f}")
            
            # If stored with correct prefix, search_query should match better
            prefix_works = sim_with_prefix > 0.85
            print(f"\n   {check_mark(prefix_works)} Embedding prefix working correctly", end="")
            if prefix_works:
                print(f" (similarity > 0.85)")
            else:
                print(f" - vectors may need re-ingestion!")
                all_passed = False
    
    # ─────────────────────────────────────────────────────────────
    # 4. TEST RETRIEVAL QUALITY
    # ─────────────────────────────────────────────────────────────
    print("\n🎯 4. RETRIEVAL QUALITY TEST")
    print("-" * 40)
    
    # Use text from actual chunk as query (should get high similarity)
    if points:
        test_text = points[0].payload.get("text", "")
        test_query = test_text[:80]  # Use beginning of a real chunk
        
        print(f"   Test query: '{test_query[:50]}...'")
        
        # Generate query embedding
        query_vector = await embedder.embed_query(test_query)
        
        # Search
        results = client.query_points(
            collection_name="meeting_transcripts",
            query=query_vector,
            limit=5,
            with_payload=True
        ).points
        
        if results:
            top_score = results[0].score
            print(f"\n   Top 5 retrieval scores:")
            for i, r in enumerate(results):
                print(f"      {i+1}. Score: {r.score:.4f}")
            
            # Check if top result has good similarity
            good_retrieval = top_score > 0.75
            print(f"\n   {check_mark(good_retrieval)} Top score > 0.75 threshold", end="")
            if good_retrieval:
                print(f" ({top_score:.4f})")
            else:
                print(f" ({top_score:.4f}) - retrieval may be poor")
                all_passed = False
    
    # ─────────────────────────────────────────────────────────────
    # 5. CHECK CHUNK QUALITY
    # ─────────────────────────────────────────────────────────────
    print("\n📏 5. CHUNK SIZE ANALYSIS")
    print("-" * 40)
    
    # Sample multiple chunks
    sample_points = client.scroll(
        collection_name="meeting_transcripts",
        limit=20,
        with_payload=True,
        with_vectors=False
    )[0]
    
    if sample_points:
        chunk_lengths = []
        for p in sample_points:
            text = p.payload.get("text", "")
            word_count = len(text.split())
            chunk_lengths.append(word_count)
        
        avg_words = sum(chunk_lengths) / len(chunk_lengths)
        min_words = min(chunk_lengths)
        max_words = max(chunk_lengths)
        
        print(f"   Sample of {len(chunk_lengths)} chunks:")
        print(f"   Average words per chunk: {avg_words:.0f}")
        print(f"   Min words: {min_words}")
        print(f"   Max words: {max_words}")
        
        # Good chunks should have 50-500 words
        good_size = 50 <= avg_words <= 500
        print(f"\n   {check_mark(good_size)} Chunk size in healthy range (50-500 words)")
        if not good_size:
            all_passed = False
    
    # ─────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ingestion is working correctly!")
    else:
        print("❌ SOME CHECKS FAILED - Run 'python scripts/reingest_all.py'")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(verify_ingestion())
    sys.exit(0 if result else 1)
