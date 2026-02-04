"""
Background ingestion task that runs on server startup.

Features:
- Runs automatically when FastAPI starts
- Continues from where it left off (tracks progress in file)
- Skips already-ingested transcripts
- Non-blocking (runs in background)
- Rate limit aware with exponential backoff
- Incremental sync mode (only fetches new transcripts after initial sync)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from qdrant_client.http import models as qm

from app.core.config import settings
from app.ingestion.loaders.fireflies_loader import FirefliesLoader
from app.ingestion.chunker import TokenChunker
from app.embeddings.factory import get_embedding_client
from app.vectorstore.qdrant import QdrantStore
from app.utils.id_generator import stable_point_id

logger = logging.getLogger(__name__)

# Progress tracking file location
PROGRESS_FILE = Path(__file__).parent.parent.parent / "data" / "ingestion_progress.json"

# Configuration
WAVE_SIZE = 50  # Transcripts per wave
UPSERT_BATCH = 100  # Upsert frequency
SYNC_INTERVAL_MINUTES = 30  # Re-sync interval
RATE_LIMIT_DELAY = 2.0  # Delay between API calls (seconds)
MAX_CONSECUTIVE_SKIPS = 200  # Stop scanning after this many consecutive skips (all ingested)


class IngestionProgress:
    """Tracks and persists ingestion progress."""
    
    def __init__(self, progress_file: Path = PROGRESS_FILE):
        self.progress_file = progress_file
        self._ensure_data_dir()
        self._data = self._load()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load(self) -> Dict[str, Any]:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
        
        return {
            "last_sync": None,
            "total_ingested": 0,
            "total_skipped": 0,
            "ingested_ids": [],
            "last_error": None,
            "sync_in_progress": False,
            "initial_sync_complete": False,  # Track if we've done full initial sync
        }
    
    def _save(self):
        """Save progress to file."""
        try:
            with open(self.progress_file, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress file: {e}")
    
    @property
    def last_sync(self) -> Optional[str]:
        return self._data.get("last_sync")
    
    @property
    def total_ingested(self) -> int:
        return self._data.get("total_ingested", 0)
    
    @property
    def ingested_ids(self) -> set:
        return set(self._data.get("ingested_ids", []))
    
    @property
    def sync_in_progress(self) -> bool:
        return self._data.get("sync_in_progress", False)
    
    @property
    def initial_sync_complete(self) -> bool:
        """Check if initial full sync has been completed."""
        return self._data.get("initial_sync_complete", False)
    
    def mark_initial_sync_complete(self):
        """Mark that initial sync is complete - future syncs will be incremental."""
        self._data["initial_sync_complete"] = True
        self._save()
    
    def mark_ingested(self, transcript_id: str):
        """Mark a transcript as ingested."""
        if transcript_id not in self._data.get("ingested_ids", []):
            self._data.setdefault("ingested_ids", []).append(transcript_id)
            self._data["total_ingested"] = self._data.get("total_ingested", 0) + 1
        self._save()
    
    def mark_skipped(self):
        """Increment skipped count."""
        self._data["total_skipped"] = self._data.get("total_skipped", 0) + 1
    
    def start_sync(self):
        """Mark sync as started."""
        self._data["sync_in_progress"] = True
        self._save()
    
    def end_sync(self, error: Optional[str] = None):
        """Mark sync as completed."""
        self._data["sync_in_progress"] = False
        self._data["last_sync"] = datetime.utcnow().isoformat()
        self._data["last_error"] = error
        self._save()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        return {
            "last_sync": self._data.get("last_sync"),
            "total_ingested": self._data.get("total_ingested", 0),
            "total_skipped": self._data.get("total_skipped", 0),
            "unique_transcripts": len(self._data.get("ingested_ids", [])),
            "sync_in_progress": self._data.get("sync_in_progress", False),
            "last_error": self._data.get("last_error"),
        }


class BackgroundIngestionService:
    """
    Background service for automatic Fireflies ingestion.
    """
    
    def __init__(self):
        self.loader = FirefliesLoader()
        self.embedder = get_embedding_client()
        self.store = QdrantStore()
        self.chunker = TokenChunker(chunk_size=512, overlap=64)
        self.progress = IngestionProgress()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start background ingestion."""
        if self._running:
            logger.info("Background ingestion already running")
            return
        
        # Check if Fireflies API key is configured
        if settings.FIREFLIES_API_KEY == "__MISSING__":
            logger.warning("Fireflies API key not configured - background ingestion disabled")
            return
        
        # Reset sync progress flag so ingestion runs on startup
        self.progress.end_sync("Service restarted")
        
        self._running = True
        self._task = asyncio.create_task(self._run_sync_loop())
        logger.info("Background ingestion service started")
    
    async def stop(self):
        """Stop background ingestion."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Background ingestion service stopped")
    
    async def _run_sync_loop(self):
        """Main sync loop that runs periodically."""
        while self._running:
            try:
                await self._sync_transcripts()
            except Exception as e:
                error_str = str(e).lower()
                if "502" in error_str or "bad gateway" in error_str:
                    logger.error(f"Sync failed: Fireflies server temporarily unavailable (502 Bad Gateway): {e}")
                elif "503" in error_str or "service unavailable" in error_str:
                    logger.error(f"Sync failed: Fireflies service unavailable (503): {e}")
                elif "504" in error_str or "gateway timeout" in error_str:
                    logger.error(f"Sync failed: Fireflies gateway timeout (504): {e}")
                else:
                    logger.error(f"Background sync failed: {e}")
                self.progress.end_sync(str(e))
            
            # Wait before next sync
            if self._running:
                logger.info(f"Next sync in {SYNC_INTERVAL_MINUTES} minutes")
                await asyncio.sleep(SYNC_INTERVAL_MINUTES * 60)
    
    async def _sync_transcripts(self):
        """Perform a sync of Fireflies transcripts (incremental after initial sync)."""
        if self.progress.sync_in_progress:
            logger.warning("Sync already in progress, skipping")
            return
        
        self.progress.start_sync()
        
        # Determine sync mode
        is_incremental = self.progress.initial_sync_complete
        if is_incremental:
            logger.info("🚀 Starting incremental Fireflies sync (checking for new transcripts only)...")
        else:
            logger.info("🚀 Starting full Fireflies sync...")
        
        # Ensure collection exists
        vector_size = 768  # nomic-embed-text
        self.store.ensure_collection(settings.QDRANT_COLLECTION_TRANSCRIPTS, vector_size)
        
        skip = 0
        wave_num = 1
        total_processed = 0
        total_ingested = 0
        total_skipped = 0
        consecutive_skips = 0  # Track consecutive skips for early termination
        points_buffer: list[qm.PointStruct] = []
        rate_limit_backoff = RATE_LIMIT_DELAY
        
        try:
            while True:
                logger.info(f"🌊 Wave {wave_num}: Fetching {WAVE_SIZE} transcripts (skip={skip})...")
                
                try:
                    batch = await self.loader.list_transcripts(limit=WAVE_SIZE, skip=skip)
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "too_many_requests" in error_str:
                        # Rate limited on list - wait and retry
                        logger.warning(f"⏳ Rate limited on list, waiting {rate_limit_backoff}s...")
                        await asyncio.sleep(rate_limit_backoff)
                        rate_limit_backoff = min(rate_limit_backoff * 2, 120)  # Exponential backoff, max 2 min
                        continue
                    elif "502" in error_str or "bad gateway" in error_str:
                        # Server error - log and let sync loop handle retry
                        logger.error(f"🔴 Fireflies server error (502 Bad Gateway): {e}")
                        raise  # Re-raise to trigger sync loop retry
                    elif "503" in error_str or "service unavailable" in error_str:
                        logger.error(f"🔴 Fireflies service unavailable (503): {e}")
                        raise
                    elif "504" in error_str or "gateway timeout" in error_str:
                        logger.error(f"🔴 Fireflies gateway timeout (504): {e}")
                        raise
                    else:
                        # Other errors
                        logger.error(f"🔴 Fireflies API error: {e}")
                        raise
                
                if not batch:
                    logger.info("   No more transcripts to fetch")
                    break
                
                logger.info(f"   Fetched {len(batch)} transcripts")
                
                for t in batch:
                    total_processed += 1
                    transcript_id = t["id"]
                    source_file = f"fireflies:{transcript_id}"
                    
                    # Skip if already in progress tracker OR in Qdrant
                    if transcript_id in self.progress.ingested_ids:
                        logger.debug(f"[{total_processed}] ⏭️  {t.get('title', transcript_id)[:40]} (in progress file)")
                        total_skipped += 1
                        consecutive_skips += 1
                        continue
                    
                    if self.store.check_source_exists(settings.QDRANT_COLLECTION_TRANSCRIPTS, source_file):
                        logger.debug(f"[{total_processed}] ⏭️  {t.get('title', transcript_id)[:40]} (in Qdrant)")
                        self.progress.mark_ingested(transcript_id)  # Add to progress
                        total_skipped += 1
                        consecutive_skips += 1
                        continue
                    
                    # Reset consecutive skips - we found something new
                    consecutive_skips = 0
                    
                    # Add rate limit delay before API calls
                    await asyncio.sleep(rate_limit_backoff)
                    
                    try:
                        transcript = await self.loader.get_transcript(transcript_id)
                        sentences = transcript.get("sentences", [])
                        
                        # Reset backoff on success
                        rate_limit_backoff = RATE_LIMIT_DELAY
                        
                        if not sentences:
                            logger.debug(f"[{total_processed}] ⚠️  {t.get('title', transcript_id)[:40]} (no sentences)")
                            total_skipped += 1
                            continue
                        
                        # Chunk with speaker awareness
                        base_metadata = {
                            "transcript_id": transcript_id,
                            "title": transcript.get("title"),
                            "date": transcript.get("date"),
                            "organizer_email": transcript.get("organizer_email"),
                            "source_file": source_file,
                        }
                        
                        chunked_records = self.chunker.chunk_with_speakers(sentences, base_metadata)
                        texts = [c["text"] for c in chunked_records]
                        
                        # Batch embed
                        vectors = await self.embedder.embed_batch(texts)
                        
                        # Build points
                        for idx, (rec, vector) in enumerate(zip(chunked_records, vectors)):
                            payload = {
                                **rec["metadata"],
                                "text": rec["text"],
                                "chunk_index": idx,
                            }
                            
                            section_id = rec["metadata"].get("chunk_index", idx)
                            point_id = stable_point_id("fireflies", transcript_id, str(section_id), str(idx))
                            
                            points_buffer.append(
                                qm.PointStruct(id=point_id, vector=vector, payload=payload)
                            )
                        
                        logger.info(f"[{total_processed}] ✅ {t.get('title', transcript_id)[:40]} ({len(chunked_records)} chunks)")
                        self.progress.mark_ingested(transcript_id)
                        total_ingested += 1
                        
                        # Periodic upsert
                        if total_ingested % UPSERT_BATCH == 0 and points_buffer:
                            logger.info(f"\n💾 Upserting {len(points_buffer)} points to Qdrant...")
                            self.store.upsert(settings.QDRANT_COLLECTION_TRANSCRIPTS, points_buffer)
                            points_buffer = []
                            
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "too_many_requests" in error_str.lower():
                            # Rate limited - increase backoff and skip this one for now
                            rate_limit_backoff = min(rate_limit_backoff * 2, 120)
                            logger.warning(f"[{total_processed}] ⏳ Rate limited on {t.get('title', transcript_id)[:30]}, will retry next sync")
                        elif "object_not_found" in error_str.lower() or "404" in error_str:
                            # Transcript deleted/unavailable - mark as ingested to skip it
                            logger.warning(f"[{total_processed}] ⚠️ {t.get('title', transcript_id)[:40]}: Not found (deleted?), skipping")
                            self.progress.mark_ingested(transcript_id)
                        else:
                            logger.error(f"[{total_processed}] ❌ {t.get('title', transcript_id)[:40]}: {e}")
                        total_skipped += 1
                
                # For incremental sync: stop early if we've seen many consecutive already-ingested items
                if is_incremental and consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    logger.info(f"   📊 {consecutive_skips} consecutive skips - likely caught up, stopping early")
                    break
                
                skip += WAVE_SIZE
                wave_num += 1
                
                if len(batch) < WAVE_SIZE:
                    break
            
            # Final upsert
            if points_buffer:
                logger.info(f"\n💾 Final upsert: {len(points_buffer)} points to Qdrant...")
                self.store.upsert(settings.QDRANT_COLLECTION_TRANSCRIPTS, points_buffer)
            
            # Mark initial sync complete if this was a full sync
            if not is_incremental and total_processed > 0:
                self.progress.mark_initial_sync_complete()
                logger.info("   📌 Initial sync marked complete - future syncs will be incremental")
            
            self.progress.end_sync()
            logger.info(f"\n✅ Background sync complete!")
            logger.info(f"   Total processed: {total_processed}")
            logger.info(f"   Ingested: {total_ingested}")
            logger.info(f"   Skipped: {total_skipped}")
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.progress.end_sync(str(e))
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current ingestion status."""
        return {
            "running": self._running,
            "progress": self.progress.get_stats(),
        }


# Global instance
background_ingestion = BackgroundIngestionService()
