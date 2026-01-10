"""
Integration tests with real Qdrant and Ollama using testcontainers.

These tests spin up actual containers and test the full RAG pipeline.
Run with: pytest tests/integration/ -v --integration
"""

import pytest
import asyncio
import logging
from typing import Generator, AsyncGenerator
from unittest.mock import patch

logger = logging.getLogger(__name__)

# Check if testcontainers is available
try:
    from testcontainers.qdrant import QdrantContainer
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    logger.warning("testcontainers not installed. Integration tests will be skipped.")


# Skip all tests in this module if testcontainers is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TESTCONTAINERS_AVAILABLE,
        reason="testcontainers not installed"
    ),
]


@pytest.fixture(scope="module")
def qdrant_container() -> Generator:
    """Start Qdrant container for integration tests."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not installed")
    
    container = QdrantContainer("qdrant/qdrant:latest")
    container.start()
    
    yield container
    
    container.stop()


@pytest.fixture(scope="module")
def qdrant_url(qdrant_container) -> str:
    """Get Qdrant URL from container."""
    host = qdrant_container.get_container_host_ip()
    port = qdrant_container.get_exposed_port(6333)
    return f"http://{host}:{port}"


@pytest.fixture
def mock_settings(qdrant_url: str):
    """Mock settings to use test containers."""
    with patch("app.core.config.settings") as mock:
        mock.QDRANT_URL = qdrant_url
        mock.QDRANT_COLLECTION_TRANSCRIPTS = "test_transcripts"
        mock.QDRANT_COLLECTION_DOCUMENTS = "test_documents"
        mock.EMBEDDING_MODEL = "nomic-embed-text"
        mock.LLM_MODEL = "llama3.2:3b"
        mock.OLLAMA_BASE_URL = "http://localhost:11434"
        mock.MAX_QUERY_LENGTH = 2000
        mock.SEMANTIC_CACHE_ENABLED = True
        mock.HYBRID_SEARCH_ENABLED = True
        mock.ADAPTIVE_RETRIEVAL_ENABLED = True
        mock.RERANKING_ENABLED = True
        yield mock


class TestQdrantIntegration:
    """Integration tests for Qdrant vector store."""
    
    def test_qdrant_connection(self, qdrant_url: str):
        """Test that Qdrant container is accessible."""
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections()
        
        assert collections is not None
    
    def test_create_collection(self, qdrant_url: str):
        """Test creating a collection in Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
        
        client = QdrantClient(url=qdrant_url)
        
        # Create test collection
        client.recreate_collection(
            collection_name="test_collection",
            vectors_config=qm.VectorParams(
                size=768,  # nomic-embed-text dimension
                distance=qm.Distance.COSINE,
            ),
        )
        
        # Verify collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        assert "test_collection" in collection_names
    
    def test_upsert_and_search(self, qdrant_url: str):
        """Test upserting vectors and searching."""
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
        import numpy as np
        
        client = QdrantClient(url=qdrant_url)
        
        # Create collection
        client.recreate_collection(
            collection_name="test_search",
            vectors_config=qm.VectorParams(
                size=128,
                distance=qm.Distance.COSINE,
            ),
        )
        
        # Create test vectors
        vectors = [
            np.random.rand(128).tolist() for _ in range(10)
        ]
        
        # Upsert points
        points = [
            qm.PointStruct(
                id=i + 1,  # Qdrant requires positive integers (not 0) or UUIDs
                vector=vectors[i],
                payload={"text": f"Document {i}", "index": i},
            )
            for i in range(10)
        ]
        
        client.upsert(
            collection_name="test_search",
            points=points,
        )
        
        # Search using query_points (newer API)
        results = client.query_points(
            collection_name="test_search",
            query=vectors[0],
            limit=5,
        ).points
        
        assert len(results) == 5
        assert results[0].id == 1  # Most similar should be itself (id starts at 1)
        assert results[0].score > 0.99


class TestQdrantStoreIntegration:
    """Integration tests for QdrantStore class."""
    
    def test_qdrant_store_ensure_collection(self, qdrant_url: str, mock_settings):
        """Test QdrantStore.ensure_collection."""
        mock_settings.QDRANT_URL = qdrant_url
        
        from app.vectorstore.qdrant import QdrantStore
        
        # Create store with mocked URL
        with patch("app.vectorstore.qdrant.settings") as store_settings:
            store_settings.QDRANT_URL = qdrant_url
            store = QdrantStore()
        
        # Ensure collection
        store.ensure_collection("integration_test", vector_size=768)
        
        # Verify
        collections = store.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        assert "integration_test" in collection_names
    
    def test_qdrant_store_upsert(self, qdrant_url: str, mock_settings):
        """Test QdrantStore.upsert."""
        mock_settings.QDRANT_URL = qdrant_url
        
        from app.vectorstore.qdrant import QdrantStore
        from qdrant_client.http import models as qm
        import numpy as np
        
        with patch("app.vectorstore.qdrant.settings") as store_settings:
            store_settings.QDRANT_URL = qdrant_url
            store = QdrantStore()
        
        # Ensure collection
        store.ensure_collection("upsert_test", vector_size=128)
        
        # Create points
        points = [
            qm.PointStruct(
                id=i + 1,  # Qdrant requires positive integers (not 0) or UUIDs
                vector=np.random.rand(128).tolist(),
                payload={"text": f"Test document {i}"},
            )
            for i in range(5)
        ]
        
        # Upsert
        store.upsert("upsert_test", points)
        
        # Verify count
        info = store.client.get_collection("upsert_test")
        assert info.points_count == 5


class TestFullRAGPipelineIntegration:
    """
    Full RAG pipeline integration tests.
    
    NOTE: These tests require Ollama to be running locally.
    They will be skipped if Ollama is not available.
    """
    
    @pytest.fixture
    def check_ollama_available(self):
        """Check if Ollama is running."""
        import httpx
        
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama not responding")
        except Exception:
            pytest.skip("Ollama not available at localhost:11434")
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, check_ollama_available):
        """Test embedding generation with real Ollama."""
        from app.embeddings.client import OllamaEmbeddingClient
        
        client = OllamaEmbeddingClient()
        
        # Generate embedding
        embedding = await client.embed_one("What was discussed in the meeting?")
        
        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)
    
    @pytest.mark.asyncio
    async def test_full_rag_flow(self, qdrant_url: str, check_ollama_available):
        """Test full RAG flow: embed -> store -> retrieve -> generate."""
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
        from app.embeddings.client import OllamaEmbeddingClient
        from app.llm.ollama import OllamaClient
        
        # Setup
        qdrant = QdrantClient(url=qdrant_url)
        embedder = OllamaEmbeddingClient()
        llm = OllamaClient()
        
        # Create collection
        qdrant.recreate_collection(
            collection_name="rag_test",
            vectors_config=qm.VectorParams(
                size=768,
                distance=qm.Distance.COSINE,
            ),
        )
        
        # Create test documents
        test_docs = [
            "The Q4 revenue target was set at $10 million during the board meeting.",
            "Marketing budget was increased by 20% for the new product launch.",
            "The engineering team will deliver the MVP by March 15th.",
        ]
        
        # Embed and store documents
        for i, doc in enumerate(test_docs):
            embedding = await embedder.embed_one(doc)
            
            qdrant.upsert(
                collection_name="rag_test",
                points=[
                    qm.PointStruct(
                        id=i + 1,  # Qdrant requires positive integers (not 0) or UUIDs
                        vector=embedding,
                        payload={
                            "text": doc,
                            "title": "Board Meeting Q4",
                            "date": "2024-12-15",
                        },
                    )
                ],
            )
        
        # Query
        query = "What was the revenue target?"
        query_embedding = await embedder.embed_one(query)
        
        # Retrieve using query_points (newer API)
        results = qdrant.query_points(
            collection_name="rag_test",
            query=query_embedding,
            limit=2,
        ).points
        
        assert len(results) > 0
        assert "revenue" in results[0].payload["text"].lower()
        
        # Generate response
        context = "\n".join([r.payload["text"] for r in results])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        response = await llm.generate(prompt)
        
        assert response is not None
        assert len(response) > 0
        # Should mention the revenue target
        assert "10" in response or "million" in response.lower()
