from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qm

from app.core.config import settings
from app.core.metrics import metrics
from app.embeddings.factory import get_embedding_client
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.loaders.fireflies_loader import FirefliesLoader
from app.vectorstore.qdrant import QdrantStore
from app.schemas.ingest import FirefliesSyncResponse, IngestionStatus
from app.utils.id_generator import stable_point_id

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self) -> None:
        logger.info("Initializing IngestionService")
        self.loader = FirefliesLoader()
        self.pipeline = IngestionPipeline()
        self.embedder = get_embedding_client()
        self.store = QdrantStore()
        self.collection = settings.QDRANT_COLLECTION_TRANSCRIPTS

    async def ingest_transcript(self, transcript_id: str) -> Dict[str, Any]:
        import time
        start_time = time.time()
        
        logger.info(f"Starting ingestion for transcript: {transcript_id}")
        
        # Check if already being processed (basic duplicate prevention)
        # This is a simple check - for production, consider Redis-based locking
        processing_key = f"processing:{transcript_id}"
        if hasattr(self, '_processing_cache') and processing_key in self._processing_cache:
            logger.info(f"Transcript {transcript_id} already being processed, skipping")
            return {"transcript_id": transcript_id, "ingested": 0, "skipped": 1, "reason": "already_processing"}
        
        # Mark as processing
        if not hasattr(self, '_processing_cache'):
            self._processing_cache = set()
        self._processing_cache.add(processing_key)
        try:
            transcript = await self.loader.get_transcript(transcript_id)
        except Exception as e:
            logger.error(f"Failed to fetch transcript {transcript_id}: {e}")
            # Clean up processing cache on error
            if hasattr(self, '_processing_cache'):
                self._processing_cache.discard(processing_key)
            raise RuntimeError(f"Failed to fetch transcript: {e}") from e

        # 2) normalize + chunk (your existing pipeline expects "meeting-like" dict;
        # we're feeding it transcript dict, which is fine as long as normalizer uses fields that exist.)
        chunks = self.pipeline.process_fireflies_meeting(transcript)
        if not chunks:
            logger.warning(f"No chunks generated for transcript {transcript_id}")
            return {"transcript_id": transcript_id, "ingested": 0, "skipped": 0}
        
        # Check chunk limit with better memory management
        max_chunks = min(settings.MAX_CHUNKS_PER_TRANSCRIPT, 500)  # Cap at 500 for memory
        if len(chunks) > max_chunks:
            logger.warning(
                f"Transcript {transcript_id} has {len(chunks)} chunks, "
                f"truncating to {max_chunks} for memory efficiency"
            )
            chunks = chunks[:max_chunks]

        logger.info(f"Processing {len(chunks)} chunks for transcript {transcript_id}")

        # 3) embed chunks with contextual enrichment and proper prefix
        try:
            # Enrich text with metadata for better semantic matching
            texts = []
            for c in chunks:
                metadata = c.get("metadata", {})
                title = transcript.get("title", "Unknown meeting")
                speaker = metadata.get("speaker", "Unknown")
                # Include context in the text for better embedding quality
                enriched_text = f"Meeting: {title}. Speaker: {speaker}. Content: {c['text']}"
                texts.append(enriched_text)
            
            # Use embed_documents() which adds the search_document: prefix for nomic-embed-text
            vectors = await self.embedder.embed_documents(texts, batch_size=32)  # Increased batch size
            vector_size = len(vectors[0])
        except Exception as e:
            logger.error(f"Embedding failed for transcript {transcript_id}: {e}")
            raise RuntimeError(f"Embedding failed: {e}") from e

        # 4) ensure qdrant collection exists
        try:
            self.store.ensure_collection(self.collection, vector_size)
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise RuntimeError(f"Vector store error: {e}") from e

        # 5) upsert points (deterministic IDs => re-ingest is safe)
        points: List[qm.PointStruct] = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            # Build payload (metadata)
            metadata = c.get("metadata", {})
            
            # Get speakers list - handle both "speaker" (singular) and "speakers" (list)
            speakers_list = metadata.get("speakers", [])
            if not speakers_list and metadata.get("speaker"):
                speakers_list = [metadata.get("speaker")]
            
            payload = {
                "source": "fireflies",
                "doc_type": "meeting",  # Document type for filtering
                "transcript_id": transcript_id,
                "title": transcript.get("title"),
                "date": transcript.get("date"),  # Fireflies uses epoch ms
                "duration": transcript.get("duration"),
                "organizer_email": transcript.get("organizer_email"),
                "speakers": speakers_list,  # List of speakers for filtering
                "section_id": metadata.get("section_id"),
                "token_count": metadata.get("token_count"),
                "chunk_index": i,
                "text": c.get("text"),
            }

            pid = stable_point_id("fireflies", transcript_id, str(metadata.get("section_id")), str(i))
            points.append(qm.PointStruct(id=pid, vector=v, payload=payload))

        try:
            self.store.upsert(self.collection, points)
        except Exception as e:
            logger.error(f"Failed to upsert points for transcript {transcript_id}: {e}")
            raise RuntimeError(f"Failed to store vectors: {e}") from e
        
        logger.info(f"Successfully ingested transcript {transcript_id}: {len(points)} chunks")

        # Clean up processing cache
        if hasattr(self, '_processing_cache'):
            self._processing_cache.discard(processing_key)

        # Calculate timing and metrics
        end_time = time.time()
        processing_time = end_time - start_time
        chunks_per_second = len(points) / processing_time if processing_time > 0 else 0

        logger.info(f"Ingestion metrics for {transcript_id}: time={processing_time:.2f}s, chunks={len(points)}, rate={chunks_per_second:.2f} chunks/sec")

        return {
            "transcript_id": transcript_id,
            "ingested": len(points),
            "collection": self.collection,
            "vector_size": vector_size,
            "processing_time_seconds": processing_time,
            "chunks_per_second": chunks_per_second,
        }

    async def ingest_recent_transcripts(self, limit: int = 10) -> Dict[str, Any]:
        # Uses your working Fireflies query: transcripts(limit: Int!)
        transcripts = await self.loader.list_transcripts(limit=limit)

        results = []
        for t in transcripts:
            tid = t["id"]
            results.append(await self.ingest_transcript(tid))

        return {"requested": limit, "results": results}
    
    async def sync_transcripts(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        force_reingest: bool = False,
    ) -> FirefliesSyncResponse:
        """
        Sync transcripts from Fireflies with date filtering and deduplication.
        """
        logger.info(f"Starting Fireflies sync (from={from_date}, to={to_date}, force={force_reingest})")
        
        all_transcripts = []
        skip = 0
        batch_size = min(50, settings.MAX_BATCH_SIZE)
        
        # Fetch all transcripts with pagination and date filtering
        while True:
            batch = await self.loader.list_transcripts(
                limit=batch_size,
                skip=skip,
                from_date=from_date,
                to_date=to_date,
            )
            if not batch:
                break
            all_transcripts.extend(batch)
            skip += batch_size
            if len(batch) < batch_size:
                break
        
        transcripts_ingested = 0
        transcripts_skipped = 0
        total_chunks = 0
        
        for transcript in all_transcripts:
            transcript_id = transcript["id"]
            source_file = f"fireflies:{transcript_id}"
            
            # Check if already ingested (unless force_reingest)
            if not force_reingest and self.store.check_source_exists(self.collection, source_file):
                transcripts_skipped += 1
                continue
            
            # Ingest the transcript
            try:
                result = await self.ingest_transcript(transcript_id)
                transcripts_ingested += 1
                total_chunks += result.get("ingested", 0) if result else 0
            except Exception as e:
                # Log error but continue with other transcripts
                logger.error(f"Error ingesting {transcript_id}: {e}")
                transcripts_skipped += 1
        
        logger.info(
            f"Sync completed: found={len(all_transcripts)}, "
            f"ingested={transcripts_ingested}, skipped={transcripts_skipped}, "
            f"chunks={total_chunks}"
        )
        
        return FirefliesSyncResponse(
            status="completed",
            transcripts_found=len(all_transcripts),
            transcripts_ingested=transcripts_ingested,
            transcripts_skipped=transcripts_skipped,
            chunks_created=total_chunks,
        )
    
    async def get_status(self) -> IngestionStatus:
        """
        Get current ingestion status from Qdrant collection.
        """
        try:
            collection_info = self.store.client.get_collection(self.collection)
            total_chunks = collection_info.points_count
            
            # Get unique source_files count (transcripts)
            # This is an approximation - we scroll through and count unique source_files
            unique_sources = set()
            offset = None
            
            while True:
                result, offset = self.store.client.scroll(
                    collection_name=self.collection,
                    limit=100,
                    offset=offset,
                    with_payload=["source_file"],
                )
                
                for point in result:
                    source_file = point.payload.get("source_file")
                    if source_file:
                        unique_sources.add(source_file)
                
                if offset is None:
                    break
            
            return IngestionStatus(
                total_transcripts=len(unique_sources),
                total_chunks=total_chunks or 0,
                collection_name=self.collection,
            )
        except Exception:
            return IngestionStatus(
                total_transcripts=0,
                total_chunks=0,
                collection_name=self.collection,
            )
