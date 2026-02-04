from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenRouterEmbeddingClient:
    """
    OpenRouter embedding client for fast, hosted embeddings.

    Uses OpenAI's text-embedding-3-small model via OpenRouter for:
    - Fast embedding generation (0.2-0.5s vs 10s+ with remote Ollama)
    - High quality embeddings
    - Reliable cloud infrastructure
    """

    # Prefix constants for asymmetric search (compatible with nomic-embed-text pattern)
    QUERY_PREFIX = "search_query: "
    DOCUMENT_PREFIX = "search_document: "

    def __init__(self) -> None:
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"
        # Use OpenAI's text-embedding-3-small via OpenRouter
        # Configure to output 768 dimensions to match nomic-embed-text
        self.model = "openai/text-embedding-3-small"
        self.dimensions = 768  # Match nomic-embed-text dimensions

        # Timeouts optimized for fast API
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,  # Much faster than Ollama
            write=10.0,
            pool=10.0,
        )

        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )

    def _get_headers(self) -> dict:
        """Get headers for OpenRouter API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://dani.ai",  # Optional: for rankings
            "X-Title": "DANI Engine",  # Optional: show in OpenRouter dashboard
        }

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self.client.aclose()

    async def __aenter__(self) -> "OpenRouterEmbeddingClient":
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

    async def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Batch embed documents with the search_document prefix.
        OpenRouter/OpenAI supports up to 2048 inputs per request.
        """
        prefixed_texts = [f"{self.DOCUMENT_PREFIX}{text}" for text in texts]
        return await self.embed_batch(prefixed_texts, batch_size=batch_size)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_one(self, text: str) -> List[float]:
        """
        Embed a single text using OpenRouter.

        Returns:
            List of floats representing the embedding (1536 dimensions)
        """
        # Truncate if too long (OpenAI limit is ~8191 tokens)
        max_chars = 32000
        if len(text) > max_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
            text = text[:max_chars]

        # Clean problematic characters
        text = text.replace('\x00', '')
        text = ''.join(char for char in text if ord(char) < 65536)

        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": text,
            "dimensions": self.dimensions,  # Request 768 dimensions to match nomic-embed-text
        }

        try:
            response = await self.client.post(
                url,
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]

            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(
                f"OpenRouter embedding failed: {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"OpenRouter request error: {e}")
            raise RuntimeError(
                "Failed to connect to OpenRouter API"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in embedding: {e}")
            raise

    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Batch embedding with OpenRouter API.

        OpenRouter/OpenAI supports up to 2048 inputs per request,
        but we use smaller batches (100) for reliability.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API request (default 100)
        """
        if not texts:
            return []

        logger.info(f"Embedding batch of {len(texts)} texts (batch_size={batch_size})")
        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # OpenRouter API accepts array of inputs
                url = f"{self.base_url}/embeddings"
                payload = {
                    "model": self.model,
                    "input": batch,  # Send multiple texts at once
                    "dimensions": self.dimensions,  # Request 768 dimensions
                }

                response = await self.client.post(
                    url,
                    json=payload,
                    headers=self._get_headers()
                )
                response.raise_for_status()

                data = response.json()

                # Extract embeddings in order
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded batch {i//batch_size + 1}: {len(batch_embeddings)} texts")

            except Exception as e:
                logger.error(f"Batch embedding failed at index {i}: {e}")
                raise

            # Small delay between batches to avoid rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        logger.info(f"Successfully embedded {len(all_embeddings)} texts")
        return all_embeddings
