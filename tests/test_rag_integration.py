"""
Integration tests for the enhanced RAG pipeline.

These tests verify that all components work together correctly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService


class MockQdrantHit:
    """Mock Qdrant search hit."""
    def __init__(self, id: str, score: float, payload: Dict[str, Any]):
        self.id = id
        self.score = score
        self.payload = payload


class TestRetrievalServiceIntegration:
    """Integration tests for RetrievalService."""
    
    @pytest.fixture
    def mock_qdrant_results(self):
        """Create mock Qdrant search results."""
        return [
            MockQdrantHit(
                id="1",
                score=0.92,
                payload={
                    "text": "Alice discussed the Q3 revenue report with the team.",
                    "title": "Q3 Review Meeting",
                    "date": "2025-01-15",
                    "speakers": ["Alice", "Bob"],
                    "transcript_id": "trans_001",
                }
            ),
            MockQdrantHit(
                id="2",
                score=0.85,
                payload={
                    "text": "Bob presented the marketing strategy for next quarter.",
                    "title": "Marketing Planning",
                    "date": "2025-01-14",
                    "speakers": ["Bob", "Charlie"],
                    "transcript_id": "trans_002",
                }
            ),
            MockQdrantHit(
                id="3",
                score=0.78,
                payload={
                    "text": "The team reviewed budget allocations for Q4.",
                    "title": "Budget Review",
                    "date": "2025-01-13",
                    "speakers": ["Alice", "Dave"],
                    "transcript_id": "trans_003",
                }
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_search_with_hybrid_and_reranking(self, mock_qdrant_results):
        """Test that hybrid search and reranking work together."""
        service = RetrievalService()
        
        # Mock the dependencies
        service.store.search = AsyncMock(return_value=mock_qdrant_results)
        service.embedder.embed_one = AsyncMock(return_value=[0.1] * 768)
        
        results = await service.search(
            query="What did Alice say about revenue?",
            limit=3,
            use_hybrid=True,
            use_reranking=True,
            use_adaptive=True,
        )
        
        assert len(results) > 0
        assert all("score" in r for r in results)
        assert all("text" in r for r in results)
        
        # Results should have search_source field
        assert all("search_source" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_confidence(self, mock_qdrant_results):
        """Test search with confidence scoring."""
        service = RetrievalService()
        
        service.store.search = AsyncMock(return_value=mock_qdrant_results)
        service.embedder.embed_one = AsyncMock(return_value=[0.1] * 768)
        
        result = await service.search_with_confidence(
            query="Q3 revenue discussion",
            limit=3,
        )
        
        assert "chunks" in result
        assert "confidence" in result
        assert "query_analysis" in result
        
        # Verify confidence structure
        confidence = result["confidence"]
        assert "score" in confidence
        assert "level" in confidence
        assert "metrics" in confidence
        
        # Verify query analysis
        analysis = result["query_analysis"]
        assert "intent" in analysis
        assert "entities" in analysis
    
    @pytest.mark.asyncio
    async def test_embedding_cache_hit(self, mock_qdrant_results):
        """Test that embedding cache works correctly."""
        service = RetrievalService()
        
        service.store.search = AsyncMock(return_value=mock_qdrant_results)
        
        embed_call_count = 0
        async def mock_embed(text):
            nonlocal embed_call_count
            embed_call_count += 1
            return [0.1] * 768
        
        service.embedder.embed_one = mock_embed
        
        # First search - should call embedder
        await service.search("test query", limit=3)
        assert embed_call_count == 1
        
        # Second search with same query - should use cache
        await service.search("test query", limit=3)
        # Embed count should still be 1 (cache hit)
        assert embed_call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_qdrant_results):
        """Test cache statistics."""
        service = RetrievalService()
        
        service.store.search = AsyncMock(return_value=mock_qdrant_results)
        service.embedder.embed_one = AsyncMock(return_value=[0.1] * 768)
        
        # Perform some searches
        await service.search("query 1", limit=3)
        await service.search("query 2", limit=3)
        await service.search("query 1", limit=3)  # Cache hit
        
        stats = service.get_cache_stats()
        
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats


class TestChatServiceIntegration:
    """Integration tests for ChatService."""
    
    @pytest.fixture
    def mock_chunks(self):
        """Create mock retrieval chunks."""
        return [
            {
                "score": 0.9,
                "text": "Alice discussed the Q3 revenue, reporting $2.5M in sales.",
                "title": "Q3 Review",
                "date": "2025-01-15",
                "speakers": ["Alice", "Bob"],
                "transcript_id": "trans_001",
                "search_source": "hybrid",
            },
            {
                "score": 0.85,
                "text": "Bob mentioned the marketing budget needs to increase by 15%.",
                "title": "Budget Meeting",
                "date": "2025-01-14",
                "speakers": ["Bob"],
                "transcript_id": "trans_002",
                "search_source": "vector",
            },
        ]
    
    @pytest.mark.asyncio
    async def test_answer_with_caching(self, mock_chunks):
        """Test that response caching works."""
        service = ChatService()
        
        # Mock dependencies
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": mock_chunks,
            "confidence": {"score": 0.85, "level": "high", "metrics": {}},
            "query_analysis": {"intent": "factual", "entities": [], "time_references": [], "processed_query": "test"},
            "disclaimer": None,
        })
        service.retrieval._get_embedding = AsyncMock(return_value=[0.1] * 768)
        service.llm.generate = AsyncMock(return_value="The Q3 revenue was $2.5M.")
        
        # First call - should generate
        response1 = await service.answer("What was the Q3 revenue?", use_cache=True)
        
        assert "answer" in response1
        assert response1["cache_hit"] is False
        
        # Second call - should hit cache
        response2 = await service.answer("What was the Q3 revenue?", use_cache=True)
        
        assert response2["cache_hit"] is True
    
    @pytest.mark.asyncio
    async def test_answer_with_confidence(self, mock_chunks):
        """Test that answer includes confidence information."""
        service = ChatService()
        
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": mock_chunks,
            "confidence": {
                "score": 0.85,
                "level": "high",
                "reason": "good_retrieval",
                "metrics": {"top_similarity": 0.9, "avg_similarity": 0.875, "chunk_count": 2},
            },
            "query_analysis": {"intent": "factual", "entities": [], "time_references": [], "processed_query": "test"},
            "disclaimer": None,
        })
        service.retrieval._get_embedding = AsyncMock(return_value=[0.1] * 768)
        service.llm.generate = AsyncMock(return_value="Answer text here.")
        
        response = await service.answer("test query", use_cache=False)
        
        assert "confidence" in response
        assert response["confidence"]["level"] == "high"
        assert "query_analysis" in response
    
    @pytest.mark.asyncio
    async def test_answer_with_low_confidence_disclaimer(self, mock_chunks):
        """Test that low confidence triggers disclaimer."""
        service = ChatService()
        
        low_confidence_chunks = [{"score": 0.5, "text": "Barely relevant", **c} for c in mock_chunks]
        
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": low_confidence_chunks,
            "confidence": {
                "score": 0.45,
                "level": "low",
                "reason": "low_relevance",
                "metrics": {},
            },
            "query_analysis": {"intent": "factual", "entities": [], "time_references": [], "processed_query": "test"},
            "disclaimer": "ℹ️ I found some relevant information, but it may not fully answer your question.",
        })
        service.retrieval._get_embedding = AsyncMock(return_value=[0.1] * 768)
        service.llm.generate = AsyncMock(return_value="Limited information available.")
        
        response = await service.answer("obscure topic", use_cache=False)
        
        assert "disclaimer" in response
        assert response["disclaimer"] is not None
    
    @pytest.mark.asyncio
    async def test_source_attribution(self, mock_chunks):
        """Test that sources have proper attribution."""
        service = ChatService()
        
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": mock_chunks,
            "confidence": {"score": 0.85, "level": "high", "metrics": {}},
            "query_analysis": {"intent": "factual", "entities": [], "time_references": [], "processed_query": "test"},
            "disclaimer": None,
        })
        service.retrieval._get_embedding = AsyncMock(return_value=[0.1] * 768)
        service.llm.generate = AsyncMock(return_value="Answer")
        
        response = await service.answer("test", use_cache=False)
        
        assert "sources" in response
        sources = response["sources"]
        
        for source in sources:
            assert "title" in source
            assert "relevance_score" in source
            assert "search_source" in source  # New field from hybrid search
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_chunks):
        """Test combined cache statistics."""
        service = ChatService()
        
        stats = service.get_cache_stats()
        
        assert "response_cache" in stats
        assert "embedding_cache" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
