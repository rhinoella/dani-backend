from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client.http import models as qm

from app.core.config import settings
from app.embeddings.client import OllamaEmbeddingClient
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.loaders.fireflies_loader import FirefliesLoader
from app.vectorstore.qdrant import QdrantStore
from app.schemas.ingest import FirefliesSyncResponse, IngestionStatus

logger = logging.getLogger(__name__)


def _stable_point_id(*parts: str) -> str:
    """Generate a stable UUID5 from multiple string parts."""
    composite = ":".join(parts)
    return str(uuid5(NAMESPACE_DNS, composite))


class IngestionService:
    def __init__(self) -> None:
        logger.info("Initializing IngestionService")
        self.loader = FirefliesLoader()
        self.pipeline = IngestionPipeline()
        self.embedder = OllamaEmbeddingClient()
        self.store = QdrantStore()
        self.collection = settings.QDRANT_COLLECTION_TRANSCRIPTS

    async def ingest_transcript(self, transcript_id: str) -> Dict[str, Any]:
        logger.info(f"Starting ingestion for transcript: {transcript_id}")
        
        # 1) pull transcript (sentences) from Fireflies
        try:
            transcript = await self.loader.get_transcript(transcript_id)
        except Exception as e:
            logger.error(f"Failed to fetch transcript {transcript_id}: {e}")
            raise RuntimeError(f"Failed to fetch transcript: {e}") from e

        # 2) normalize + chunk (your existing pipeline expects "meeting-like" dict;
        # we're feeding it transcript dict, which is fine as long as normalizer uses fields that exist.)
        chunks = self.pipeline.process_fireflies_meeting(transcript)
        if not chunks:
            logger.warning(f"No chunks generated for transcript {transcript_id}")
            return {"transcript_id": transcript_id, "ingested": 0, "skipped": 0}
        
        # Check chunk limit
        if len(chunks) > settings.MAX_CHUNKS_PER_TRANSCRIPT:
            logger.warning(
                f"Transcript {transcript_id} has {len(chunks)} chunks, "
                f"truncating to {settings.MAX_CHUNKS_PER_TRANSCRIPT}"
            )
            chunks = chunks[:settings.MAX_CHUNKS_PER_TRANSCRIPT]

        logger.info(f"Processing {len(chunks)} chunks for transcript {transcript_id}")

        # 3) embed chunks
        try:
            texts = [c["text"] for c in chunks]
            vectors = await self.embedder.embed_batch(texts)
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
            payload = {
                "source": "fireflies",
                "transcript_id": transcript_id,
                "title": transcript.get("title"),
                "date": transcript.get("date"),  # Fireflies uses epoch ms
                "duration": transcript.get("duration"),
                "organizer_email": transcript.get("organizer_email"),
                "speaker": metadata.get("speaker"),
                "section_id": metadata.get("section_id"),
                "token_count": metadata.get("token_count"),
                "chunk_index": i,
                "text": c.get("text"),
            }

            pid = _stable_point_id("fireflies", transcript_id, str(metadata.get("section_id")), str(i))
            points.append(qm.PointStruct(id=pid, vector=v, payload=payload))

        try:
            self.store.upsert(self.collection, points)
        except Exception as e:
            logger.error(f"Failed to upsert points for transcript {transcript_id}: {e}")
            raise RuntimeError(f"Failed to store vectors: {e}") from e
        
        logger.info(f"Successfully ingested transcript {transcript_id}: {len(points)} chunks")

        return {
            "transcript_id": transcript_id,
            "ingested": len(points),
            "collection": self.collection,
            "vector_size": vector_size,
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
