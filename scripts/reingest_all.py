#!/usr/bin/env python3
"""
Re-ingestion script for RAG data.

After fixing the embedding prefix mismatch, ALL existing vectors need to be re-embedded
with the correct search_document: prefix for nomic-embed-text.

This script:
1. Drops existing Qdrant collections (meeting_transcripts, documents)
2. Re-ingests all Fireflies transcripts with proper embeddings
3. Reports progress and results

Usage:
    python scripts/reingest_all.py [--dry-run] [--transcripts-only] [--limit N] [--verbose]

Options:
    --dry-run           Show what would be done without making changes
    --transcripts-only  Only re-ingest transcripts (skip documents)
    --limit N           Limit to N most recent transcripts (default: all)
    --verbose, -v       Show detailed chunk information during ingestion
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.vectorstore.qdrant import QdrantStore
from app.services.ingestion_service import IngestionService
from app.embeddings.factory import get_embedding_client
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.loaders.fireflies_loader import FirefliesLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global verbose flag
VERBOSE = False


async def check_ollama_connection():
    """Verify Ollama is running and embedding model is available."""
    print("🔍 Checking Ollama connection...")
    
    embedder = get_embedding_client()
    try:
        # Test embedding with the document prefix
        test_embedding = await embedder.embed_document("Test document content")
        print(f"✅ Ollama connected. Embedding size: {len(test_embedding)}")
        print(f"   Model: {settings.EMBEDDING_MODEL}")
        print(f"   URL: {settings.OLLAMA_EMBEDDINGS_URL}")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print(f"   Make sure Ollama is running at {settings.OLLAMA_EMBEDDINGS_URL}")
        print(f"   And model '{settings.EMBEDDING_MODEL}' is pulled")
        return False
    finally:
        await embedder.close()


async def drop_collections(dry_run: bool = False):
    """Drop existing Qdrant collections."""
    store = QdrantStore()
    
    collections_to_drop = [
        settings.QDRANT_COLLECTION_TRANSCRIPTS,
        settings.QDRANT_COLLECTION_DOCUMENTS,
        "test_id_format",  # Cleanup test collection
    ]
    
    for collection_name in collections_to_drop:
        try:
            # Check if collection exists
            existing = store.client.get_collections().collections
            if any(c.name == collection_name for c in existing):
                if dry_run:
                    print(f"   [DRY RUN] Would drop collection: {collection_name}")
                else:
                    store.client.delete_collection(collection_name)
                    print(f"   ✅ Dropped collection: {collection_name}")
            else:
                print(f"   ℹ️ Collection doesn't exist: {collection_name}")
        except Exception as e:
            print(f"   ⚠️ Error with collection {collection_name}: {e}")


async def reingest_transcripts(dry_run: bool = False, limit: int = None):
    """Re-ingest all Fireflies transcripts."""
    service = IngestionService()
    loader = FirefliesLoader()
    pipeline = IngestionPipeline()
    
    print("\n📋 Fetching transcript list from Fireflies...")
    
    try:
        # Get all transcripts (paginated)
        all_transcripts = []
        skip = 0
        batch_size = 50
        
        while True:
            batch = await service.loader.list_transcripts(limit=batch_size, skip=skip)
            if not batch:
                break
            all_transcripts.extend(batch)
            skip += batch_size
            if len(batch) < batch_size:
                break
            if limit and len(all_transcripts) >= limit:
                all_transcripts = all_transcripts[:limit]
                break
        
        print(f"   Found {len(all_transcripts)} transcripts")
        
        if dry_run:
            print(f"\n[DRY RUN] Would re-ingest {len(all_transcripts)} transcripts")
            for t in all_transcripts[:5]:
                print(f"   - {t.get('title', 'Untitled')} ({t['id']})")
            if len(all_transcripts) > 5:
                print(f"   ... and {len(all_transcripts) - 5} more")
            return 0, 0
        
        print(f"\n🔄 Re-ingesting {len(all_transcripts)} transcripts...")
        
        success_count = 0
        error_count = 0
        total_chunks = 0
        
        for i, transcript in enumerate(all_transcripts, 1):
            tid = transcript['id']
            title = transcript.get('title', 'Untitled')[:50]
            
            try:
                # Show transcript header
                print(f"\n{'─' * 60}")
                print(f"📝 [{i}/{len(all_transcripts)}] {title}")
                print(f"   ID: {tid}")
                
                if VERBOSE:
                    # Fetch full transcript for verbose output
                    full_transcript = await loader.get_transcript(tid)
                    
                    # Show transcript metadata
                    date = full_transcript.get('dateString', full_transcript.get('date', 'Unknown'))
                    organizer = full_transcript.get('organizer_email', 'Unknown')
                    participants = full_transcript.get('participants', [])
                    
                    print(f"   Date: {date}")
                    print(f"   Organizer: {organizer}")
                    if participants:
                        print(f"   Participants: {', '.join(p.get('name', p.get('email', '?')) for p in participants[:5])}")
                        if len(participants) > 5:
                            print(f"                 ... and {len(participants) - 5} more")
                    
                    # Process through pipeline to see chunks
                    chunks = pipeline.process_fireflies_meeting(full_transcript)
                    
                    print(f"\n   📦 Chunks generated: {len(chunks)}")
                    print(f"   {'─' * 50}")
                    
                    for j, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
                        text = chunk.get('text', '')
                        speakers = chunk.get('speakers', [])
                        word_count = len(text.split())
                        
                        # Truncate text for display
                        display_text = text[:150] + "..." if len(text) > 150 else text
                        display_text = display_text.replace('\n', ' ')
                        
                        print(f"\n   Chunk {j+1}:")
                        print(f"      Speakers: {speakers}")
                        print(f"      Words: {word_count}")
                        print(f"      Text: \"{display_text}\"")
                    
                    if len(chunks) > 10:
                        print(f"\n   ... and {len(chunks) - 10} more chunks")
                    
                    print(f"\n   🔄 Generating embeddings with 'search_document:' prefix...")
                
                # Actually ingest
                result = await service.ingest_transcript(tid)
                chunks_count = result.get('ingested', 0)
                vector_size = result.get('vector_size', 768)
                total_chunks += chunks_count
                success_count += 1
                
                print(f"   ✅ Ingested: {chunks_count} chunks (vector dim: {vector_size})")
                
                if VERBOSE:
                    # Show sample of what was stored
                    print(f"\n   📊 Payload structure:")
                    print(f"      - transcript_id: {tid}")
                    print(f"      - doc_type: 'meeting'")
                    print(f"      - speakers: [list of speakers per chunk]")
                    print(f"      - meeting_title: {title}")
                    
            except Exception as e:
                error_count += 1
                print(f"   ❌ Error: {e}")
                if VERBOSE:
                    import traceback
                    traceback.print_exc()
        
        return success_count, total_chunks
        
    except Exception as e:
        print(f"❌ Failed to fetch transcripts: {e}")
        return 0, 0


async def main():
    global VERBOSE
    
    parser = argparse.ArgumentParser(description='Re-ingest RAG data with fixed embeddings')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--transcripts-only', action='store_true', help='Only process transcripts')
    parser.add_argument('--limit', type=int, help='Limit number of transcripts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed chunk information')
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    
    print("=" * 60)
    print("🔧 RAG Re-ingestion Script")
    print("=" * 60)
    print(f"\nThis script will re-ingest all data with FIXED embeddings.")
    print(f"The fix ensures documents use 'search_document:' prefix for nomic-embed-text.\n")
    
    if args.dry_run:
        print("🔵 DRY RUN MODE - No changes will be made\n")
    
    if VERBOSE:
        print("🔊 VERBOSE MODE - Showing detailed chunk information\n")
    
    # Step 1: Check Ollama
    if not await check_ollama_connection():
        print("\n❌ Cannot proceed without Ollama connection")
        return 1
    
    # Step 2: Confirm
    if not args.dry_run:
        print("\n⚠️  WARNING: This will DELETE all existing embeddings!")
        response = input("   Type 'yes' to continue: ")
        if response.lower() != 'yes':
            print("   Aborted.")
            return 0
    
    # Step 3: Drop collections
    print("\n📦 Step 1: Dropping existing collections...")
    await drop_collections(dry_run=args.dry_run)
    
    # Step 4: Re-ingest transcripts
    print("\n📝 Step 2: Re-ingesting transcripts...")
    success, chunks = await reingest_transcripts(
        dry_run=args.dry_run, 
        limit=args.limit
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    if args.dry_run:
        print("   [DRY RUN] No changes made")
    else:
        print(f"   ✅ Transcripts ingested: {success}")
        print(f"   📄 Total chunks created: {chunks}")
        print(f"\n🎉 Re-ingestion complete!")
        print(f"\nYour RAG system now uses proper embedding prefixes.")
        print(f"Query quality should be significantly improved.")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
