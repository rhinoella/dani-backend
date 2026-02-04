"""Embedding client factory to switch between providers."""
from __future__ import annotations

import logging
from typing import Union

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_embedding_client() -> Union["OllamaEmbeddingClient", "OpenRouterEmbeddingClient"]:
    """
    Factory function to get the appropriate embedding client based on config.

    Returns:
        OllamaEmbeddingClient or OpenRouterEmbeddingClient
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "openrouter":
        from app.embeddings.openrouter_client import OpenRouterEmbeddingClient

        if not settings.OPENROUTER_API_KEY:
            logger.error("EMBEDDING_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set")
            raise ValueError("OPENROUTER_API_KEY must be set when using OpenRouter embeddings")

        logger.info("Using OpenRouter for embeddings (fast, hosted)")
        return OpenRouterEmbeddingClient()

    elif provider == "ollama":
        from app.embeddings.client import OllamaEmbeddingClient

        logger.info("Using Ollama for embeddings (self-hosted)")
        return OllamaEmbeddingClient()

    else:
        logger.error(f"Unknown EMBEDDING_PROVIDER: {provider}")
        raise ValueError(
            f"Invalid EMBEDDING_PROVIDER: {provider}. Must be 'ollama' or 'openrouter'"
        )
