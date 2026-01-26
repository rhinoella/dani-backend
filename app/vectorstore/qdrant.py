from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.circuit_breaker import qdrant_breaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)

# Thread pool for running sync Qdrant operations without blocking event loop
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="qdrant")  # Increased from 4

# Retry decorator for Qdrant operations
qdrant_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((UnexpectedResponse, ConnectionError, TimeoutError)),
    reraise=True,
)


class QdrantStore:
    def __init__(self) -> None:
        self.url = settings.QDRANT_URL
        logger.info(f"Initializing Qdrant client at {self.url}")
        self.client = QdrantClient(url=self.url, timeout=30)

    @qdrant_retry
    def ensure_collection(self, name: str, vector_size: int) -> None:
        """Ensure collection exists, create if not."""
        try:
            existing = self.client.get_collections().collections
            if any(c.name == name for c in existing):
                logger.debug(f"Collection '{name}' already exists")
                return

            logger.info(f"Creating collection '{name}' with vector size {vector_size}")
            self.client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=vector_size,
                    distance=qm.Distance.COSINE,
                ),
            )
            
            # Create payload indexes for faster filtering
            self._create_payload_indexes(name)
            logger.info(f"Collection '{name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to ensure collection '{name}': {e}")
            raise
    
    def _create_payload_indexes(self, collection_name: str) -> None:
        """Create indexes on commonly queried payload fields for better performance."""
        indexes = [
            ("meeting_date", qm.PayloadSchemaType.INTEGER),
            ("date", qm.PayloadSchemaType.INTEGER),
            ("source_file", qm.PayloadSchemaType.KEYWORD),
            ("speakers", qm.PayloadSchemaType.KEYWORD),
            ("transcript_id", qm.PayloadSchemaType.KEYWORD),
            ("doc_type", qm.PayloadSchemaType.KEYWORD),  # For document type filtering
            ("document_id", qm.PayloadSchemaType.KEYWORD),  # For document collection
            ("organizer_email", qm.PayloadSchemaType.KEYWORD),
        ]
        
        for field_name, schema_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                # Index might already exist, skip
                pass

    @qdrant_retry
    def upsert(
        self,
        collection: str,
        points: List[qm.PointStruct],
    ) -> None:
        """Upsert points to collection with retry logic."""
        if not points:
            logger.warning("No points to upsert, skipping")
            return
            
        logger.debug(f"Upserting {len(points)} points to '{collection}'")
        try:
            self.client.upsert(
                collection_name=collection,
                points=points,
            )
            logger.info(f"Successfully upserted {len(points)} points to '{collection}'")
        except Exception as e:
            logger.error(f"Failed to upsert points to '{collection}': {e}")
            raise

    @qdrant_retry
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        filter_: Optional[qm.Filter] = None,
    ):
        """
        Semantic vector search with retry logic and circuit breaker.
        Runs in thread pool to avoid blocking async event loop.
        """
        logger.debug(f"Searching '{collection}' with limit={limit}")
        
        def _sync_search():
            return self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
                query_filter=filter_, 
            ).points
        
        try:
            # Circuit breaker protects against repeated failures
            with qdrant_breaker:
                # Run sync Qdrant operation in thread pool
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(_executor, _sync_search)
                logger.debug(f"Search returned {len(results)} results")
                return results
        except CircuitBreakerOpen as e:
            logger.warning(f"Qdrant circuit breaker open: {e}")
            raise RuntimeError(
                "The vector database is temporarily unavailable due to repeated failures. "
                f"Please try again in {e.recovery_time:.0f} seconds."
            ) from e
        except Exception as e:
            error_str = str(e)
            # Handle collection not found gracefully (returns empty results instead of error)
            if "404" in error_str or "Not Found" in error_str or "doesn't exist" in error_str.lower():
                logger.debug(f"Collection '{collection}' not found, returning empty results")
                return []
            logger.error(f"Search failed in '{collection}': {e}")
            raise
    
    @qdrant_retry
    def check_source_exists(self, collection: str, source_file: str) -> bool:
        """
        Check if a source_file already exists in the collection.
        Used for deduplication before ingestion.
        """
        try:
            filter_ = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="source_file",
                        match=qm.MatchValue(value=source_file),
                    )
                ]
            )
            
            # Search with limit 1 just to check existence
            result = self.client.scroll(
                collection_name=collection,
                scroll_filter=filter_,
                limit=1,
            )
            
            exists = len(result[0]) > 0
            logger.debug(f"Source '{source_file}' exists: {exists}")
            return exists
        except Exception as e:
            # Collection might not exist yet
            logger.warning(f"Check source exists failed (collection may not exist): {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Qdrant is reachable and healthy."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
