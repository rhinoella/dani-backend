from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """
    Ollama embedding client (local, safe, throttled).
    Designed for 8GB RAM machines.
    
    Uses nomic-embed-text prefixes for asymmetric search:
    - "search_query: " for queries
    - "search_document: " for documents during ingestion
    """
    
    # Prefix constants for nomic-embed-text asymmetric search
    QUERY_PREFIX = "search_query: "
    DOCUMENT_PREFIX = "search_document: "

    def __init__(self) -> None:
        # Increased timeouts for slow/large embeddings
        self.timeout = httpx.Timeout(
            connect=10.0,   # Increased from 5.0
            read=180.0,     # Increased from 60.0 (3 minutes)
            write=30.0,     # Increased from 10.0
            pool=30.0,      # Increased from 10.0
        )
        
        # Connection pooling to avoid TCP handshake overhead
        # NOTE: Headers are NOT set here - they are set per-request to support dynamic API key changes
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    
    def _get_headers(self) -> dict:
        """Get headers for embedding requests.
        
        Note: Embeddings always use local Ollama, which doesn't require
        authentication. Returns empty headers.
        """
        return {}
    
    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self.client.aclose()
    
    async def __aenter__(self) -> "OllamaEmbeddingClient":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query with the search_query prefix.
        Use this for all retrieval queries to improve relevance.
        """
        prefixed_query = f"{self.QUERY_PREFIX}{query}"
        return await self.embed_one(prefixed_query)
    
    async def embed_document(self, text: str) -> List[float]:
        """
        Embed a document with the search_document prefix.
        Use this during ingestion for documents/chunks.
        """
        prefixed_text = f"{self.DOCUMENT_PREFIX}{text}"
        return await self.embed_one(prefixed_text)
    
    async def embed_documents(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Batch embed documents with the search_document prefix.
        Use this during ingestion for better performance.
        """
        prefixed_texts = [f"{self.DOCUMENT_PREFIX}{text}" for text in texts]
        return await self.embed_batch(prefixed_texts, batch_size=batch_size)

    async def embed_one(self, text: str) -> List[float]:
        """
        Embed text without prefix. For most use cases, prefer:
        - embed_query() for search queries
        - embed_document() for documents during ingestion
        
        IMPORTANT: Embeddings ALWAYS use local Ollama (OLLAMA_EMBEDDINGS_URL).
        Cloud Ollama does not support the embeddings API, so we always
        route embedding requests to the local Ollama instance.
        """
        # Safety guard: embedding input too large - truncate instead of failing
        max_chars = 32000
        if len(text) > max_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
            text = text[:max_chars]
        
        # Clean problematic characters that can cause issues
        text = text.replace('\x00', '')  # Remove null bytes
        text = ''.join(char for char in text if ord(char) < 65536)  # Remove surrogate pairs

        # ALWAYS use local Ollama for embeddings (cloud doesn't support embeddings API)
        base_url = f"{settings.OLLAMA_EMBEDDINGS_URL}/api/embeddings"
        model = settings.EMBEDDING_MODEL

        payload = {
            "model": model,
            "prompt": text,
        }

        max_retries = 5  # Increased from 3
        for attempt in range(max_retries):
            try:
                response = await self.client.post(base_url, json=payload, headers=self._get_headers())

                if response.status_code == 200:
                    data = response.json()
                    return data["embedding"]

                # Log the error
                logger.warning(f"Embedding attempt {attempt + 1}/{max_retries} failed: HTTP {response.status_code}")
                
                # retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                    await asyncio.sleep(wait_time)
                    continue

                # detailed error on final failure
                try:
                    error_detail = response.json()
                    raise RuntimeError(
                        f"Ollama error {response.status_code}: {error_detail}"
                    )
                except Exception:
                    response.raise_for_status()
            
            except httpx.ReadTimeout as e:
                logger.warning(f"Embedding timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(
                    f"Ollama embedding timed out after {max_retries} attempts"
                ) from e
                    
            except httpx.ConnectError as e:
                logger.error(f"Cannot connect to Ollama for embedding: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(
                    f"Cannot connect to Ollama. Is it running at {settings.OLLAMA_BASE_URL}?"
                ) from e
            
            except Exception as e:
                logger.warning(f"Unexpected error in embedding (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise RuntimeError("Ollama embedding failed after retries")

    async def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Batch embedding with parallel processing.
        
        Processes embeddings in parallel batches for better performance.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of concurrent embedding requests (default 8, increased from 2)
                       Lower values = more reliable but slower
                       Higher values = faster but more memory/load on Ollama
        """
        if not texts:
            return []
            
        logger.info(f"Embedding batch of {len(texts)} texts (batch_size={batch_size})")
        embeddings: List[List[float]] = []

        # Process in parallel batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Retry the entire batch on failure
            max_batch_retries = 3
            for batch_attempt in range(max_batch_retries):
                try:
                    # Run batch in parallel
                    batch_results = await asyncio.gather(
                        *[self.embed_one(text) for text in batch],
                        return_exceptions=True
                    )
                    
                    # Check for errors in batch
                    has_error = False
                    for idx, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Embedding failed at index {i + idx}: {result}")
                            has_error = True
                            break
                    
                    if has_error:
                        if batch_attempt < max_batch_retries - 1:
                            wait_time = 5 * (batch_attempt + 1)  # 5, 10, 15 seconds
                            logger.info(f"Retrying batch in {wait_time}s (attempt {batch_attempt + 2}/{max_batch_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RuntimeError(
                                f"Ollama embedding failed at batch index {i} after {max_batch_retries} attempts"
                            )
                    
                    # All successful
                    embeddings.extend(batch_results)
                    break
                    
                except Exception as e:
                    if batch_attempt < max_batch_retries - 1:
                        wait_time = 5 * (batch_attempt + 1)
                        logger.info(f"Batch error, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    raise
            
            # Small delay between batches to prevent overwhelming Ollama
            await asyncio.sleep(0.3)
            
            # Log progress
            processed = min(i + batch_size, len(texts))
            if processed % 20 == 0 or processed == len(texts):
                logger.debug(f"Embedded {processed}/{len(texts)} texts")
        
        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return embeddings
