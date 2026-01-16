"""
Tests for semantic caching functionality.
"""

import pytest
import asyncio
import time
from typing import List

from app.cache.semantic_cache import (
    SemanticCache,
    ResponseCache,
    CacheEntry,
)
from app.utils.similarity import cosine_similarity


class TestCosineSimiilarity:
    """Tests for cosine similarity function."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0"""
        vec = [1.0, 2.0, 3.0, 4.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0"""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0"""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)
    
    def test_similar_vectors(self):
        """Similar vectors should have high similarity"""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity > 0.99
    
    def test_zero_vector(self):
        """Zero vector should return 0 similarity"""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_expiration_not_expired(self):
        """Fresh entry should not be expired."""
        entry = CacheEntry(
            query="test",
            query_vector=[1.0, 2.0],
            response="answer",
        )
        assert not entry.is_expired(ttl_seconds=3600)
    
    def test_expiration_expired(self):
        """Old entry should be expired."""
        entry = CacheEntry(
            query="test",
            query_vector=[1.0, 2.0],
            response="answer",
            created_at=time.time() - 7200,  # 2 hours ago
        )
        assert entry.is_expired(ttl_seconds=3600)


class TestSemanticCache:
    """Tests for SemanticCache class."""
    
    def test_exact_match_hit(self):
        """Should hit cache on exact query match."""
        cache = SemanticCache(similarity_threshold=0.95)
        
        query = "What was discussed in the meeting?"
        vector = [1.0, 2.0, 3.0, 4.0]
        response = {"answer": "Test response"}
        
        cache.set(query, vector, response)
        result = cache.get(query, vector)
        
        assert result is not None
        cached_response, similarity = result
        assert cached_response == response
        assert similarity == 1.0
    
    def test_semantic_match_hit(self):
        """Should hit cache on semantically similar query."""
        cache = SemanticCache(similarity_threshold=0.95)
        
        query1 = "Original query"
        vector1 = [1.0, 2.0, 3.0, 4.0]
        response = {"answer": "Test response"}
        
        cache.set(query1, vector1, response)
        
        # Slightly different vector (but very similar)
        query2 = "Similar query"
        vector2 = [1.01, 2.01, 3.01, 4.01]
        
        result = cache.get(query2, vector2)
        
        assert result is not None
        cached_response, similarity = result
        assert cached_response == response
        assert similarity > 0.99
    
    def test_cache_miss_different_vector(self):
        """Should miss cache on very different vectors."""
        cache = SemanticCache(similarity_threshold=0.95)
        
        query1 = "Original query"
        vector1 = [1.0, 0.0, 0.0, 0.0]
        response = {"answer": "Test response"}
        
        cache.set(query1, vector1, response)
        
        # Very different vector
        query2 = "Different query"
        vector2 = [0.0, 1.0, 0.0, 0.0]
        
        result = cache.get(query2, vector2)
        assert result is None
    
    def test_cache_eviction(self):
        """Should evict oldest entries when full."""
        cache = SemanticCache(max_size=3)
        
        for i in range(5):
            cache.set(f"query_{i}", [float(i)], f"response_{i}")
        
        # Should have max 3 entries
        assert len(cache._cache) == 3
        
        # Oldest entries should be evicted
        assert cache.get("query_0") is None
        assert cache.get("query_1") is None
        
        # Newest entries should remain
        assert cache.get("query_4") is not None
    
    def test_ttl_expiration(self):
        """Should not return expired entries."""
        cache = SemanticCache(ttl_seconds=1)
        
        query = "test query"
        vector = [1.0, 2.0]
        cache.set(query, vector, "response")
        
        # Should hit immediately
        assert cache.get(query, vector) is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should miss after expiration
        assert cache.get(query, vector) is None
    
    def test_invalidate(self):
        """Should invalidate specific entry."""
        cache = SemanticCache()
        
        cache.set("query1", [1.0], "response1")
        cache.set("query2", [2.0], "response2")
        
        assert cache.invalidate("query1")
        assert cache.get("query1") is None
        assert cache.get("query2") is not None
    
    def test_clear(self):
        """Should clear all entries."""
        cache = SemanticCache()
        
        cache.set("query1", [1.0], "response1")
        cache.set("query2", [2.0], "response2")
        
        cache.clear()
        
        assert len(cache._cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_stats(self):
        """Should track accurate statistics."""
        cache = SemanticCache(similarity_threshold=0.99)  # High threshold to avoid semantic hits
        
        cache.set("query1", [1.0, 0.0, 0.0], "response1")
        cache.get("query1", [1.0, 0.0, 0.0])  # Hit (exact match)
        cache.get("query2", [0.0, 1.0, 0.0])  # Miss (orthogonal - cosine = 0)
        cache.get("query3", [0.0, 0.0, 1.0])  # Miss (orthogonal - cosine = 0)
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["hits"] >= 1  # At least 1 hit
        assert stats["misses"] >= 2  # At least 2 misses
        assert stats["hit_rate"] > 0  # Valid hit rate


class TestResponseCache:
    """Tests for ResponseCache class."""
    
    @pytest.mark.asyncio
    async def test_get_or_generate_miss(self):
        """Should generate new response on cache miss."""
        cache = ResponseCache()
        
        async def generator():
            return {"answer": "generated response"}
        
        response, was_cached = await cache.get_or_generate(
            query="test query",
            query_vector=[1.0, 2.0, 3.0],
            generator_fn=generator,
        )
        
        assert response["answer"] == "generated response"
        assert was_cached is False
        assert response["_cache"]["hit"] is False
    
    @pytest.mark.asyncio
    async def test_get_or_generate_hit(self):
        """Should return cached response on cache hit."""
        cache = ResponseCache()
        call_count = 0
        
        async def generator():
            nonlocal call_count
            call_count += 1
            return {"answer": f"response_{call_count}"}
        
        # First call - generates
        response1, _ = await cache.get_or_generate(
            query="test query",
            query_vector=[1.0, 2.0, 3.0],
            generator_fn=generator,
        )
        
        # Second call - should hit cache
        response2, was_cached = await cache.get_or_generate(
            query="test query",
            query_vector=[1.0, 2.0, 3.0],
            generator_fn=generator,
        )
        
        assert call_count == 1  # Generator only called once
        assert response2["answer"] == "response_1"
        assert was_cached is True
        assert response2["_cache"]["hit"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
