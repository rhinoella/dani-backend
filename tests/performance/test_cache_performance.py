"""
Load and performance tests for caching effectiveness.

These tests verify:
- Cache hit rates
- Response time improvements with caching
- Memory efficiency
- Concurrent request handling

Run with: pytest tests/performance/ -v --performance
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock
import random


# Skip unless --performance flag is provided
pytestmark = pytest.mark.performance


class TestSemanticCachePerformance:
    """Performance tests for semantic caching."""
    
    def test_cache_hit_rate(self):
        """Test that cache achieves expected hit rate."""
        from app.cache.semantic_cache import SemanticCache
        import numpy as np
        
        cache = SemanticCache(
            similarity_threshold=0.95,
            max_size=100,
            ttl_seconds=3600,
        )
        
        # Generate base queries with embeddings
        base_queries = [f"What happened in meeting {i}?" for i in range(20)]
        base_vectors = [np.random.rand(768).tolist() for _ in range(20)]
        
        # Populate cache
        for query, vector in zip(base_queries, base_vectors):
            cache.set(query, vector, {"response": f"Response for {query}"})
        
        # Test with similar queries (should hit cache)
        hits = 0
        total = 100
        
        for _ in range(total):
            # Pick a random base query
            idx = random.randint(0, len(base_queries) - 1)
            query_vector = base_vectors[idx]
            
            # Add small noise to simulate similar query
            noisy_vector = [v + random.gauss(0, 0.01) for v in query_vector]
            
            result = cache.get(f"similar query {idx}", noisy_vector)
            if result is not None:
                hits += 1
        
        hit_rate = hits / total
        
        # Should have >80% hit rate with similar queries
        assert hit_rate > 0.8, f"Cache hit rate too low: {hit_rate:.2%}"
        print(f"\nCache hit rate: {hit_rate:.2%}")
    
    def test_cache_lookup_speed(self):
        """Test cache lookup performance."""
        from app.cache.semantic_cache import SemanticCache
        import numpy as np
        
        cache = SemanticCache(
            similarity_threshold=0.95,
            max_size=500,
            ttl_seconds=3600,
        )
        
        # Fill cache with entries
        vectors = []
        for i in range(500):
            vector = np.random.rand(768).tolist()
            vectors.append(vector)
            cache.set(f"query_{i}", vector, {"data": f"response_{i}"})
        
        # Measure lookup times
        lookup_times = []
        
        for _ in range(100):
            query_vector = vectors[random.randint(0, 499)]
            
            start = time.perf_counter()
            cache.get("test_query", query_vector)
            end = time.perf_counter()
            
            lookup_times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(lookup_times)
        p95_time = sorted(lookup_times)[94]
        p99_time = sorted(lookup_times)[98]
        
        print(f"\nCache lookup times (500 entries):")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  P95: {p95_time:.3f}ms")
        print(f"  P99: {p99_time:.3f}ms")
        
        # O(n) semantic similarity search is expected to be ~20-30ms with 500 entries
        # Future optimization: use FAISS or move to Redis with vector search
        assert avg_time < 50, f"Cache lookup too slow: {avg_time:.3f}ms average"
    
    def test_cache_memory_efficiency(self):
        """Test cache memory usage and eviction."""
        from app.cache.semantic_cache import SemanticCache
        import numpy as np
        import sys
        
        cache = SemanticCache(
            similarity_threshold=0.95,
            max_size=100,
            ttl_seconds=3600,
        )
        
        # Add more entries than max_size
        for i in range(200):
            vector = np.random.rand(768).tolist()
            cache.set(f"query_{i}", vector, {"data": f"response_{i}"})
        
        # Should be at max_size
        assert len(cache._cache) <= 100, f"Cache exceeded max size: {len(cache._cache)}"
        
        # Stats should reflect evictions
        stats = cache.stats()
        print(f"\nCache stats after 200 inserts (max_size=100):")
        print(f"  Size: {stats['size']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")


class TestResponseCachePerformance:
    """Performance tests for full response caching."""
    
    @pytest.mark.asyncio
    async def test_response_cache_speedup(self):
        """Test that response caching significantly speeds up repeated queries."""
        from app.cache.semantic_cache import ResponseCache
        import numpy as np
        
        cache = ResponseCache(
            similarity_threshold=0.92,
            max_size=100,
            ttl_seconds=1800,
        )
        
        # Simulate slow generator (like LLM call)
        async def slow_generator():
            await asyncio.sleep(0.1)  # 100ms simulated LLM time
            return {"answer": "Test response", "sources": []}
        
        query = "What was discussed about the budget?"
        query_vector = np.random.rand(768).tolist()
        
        # First call (cache miss - should take ~100ms)
        start = time.perf_counter()
        result1, hit1 = await cache.get_or_generate(query, query_vector, slow_generator)
        first_call_time = (time.perf_counter() - start) * 1000
        
        assert not hit1, "First call should be cache miss"
        
        # Second call (cache hit - should be fast)
        start = time.perf_counter()
        result2, hit2 = await cache.get_or_generate(query, query_vector, slow_generator)
        second_call_time = (time.perf_counter() - start) * 1000
        
        assert hit2, "Second call should be cache hit"
        assert result1 == result2, "Cached response should match original"
        
        speedup = first_call_time / second_call_time
        
        print(f"\nResponse cache speedup:")
        print(f"  First call (miss): {first_call_time:.2f}ms")
        print(f"  Second call (hit): {second_call_time:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Cache hit should be at least 10x faster
        assert speedup > 10, f"Cache speedup too low: {speedup:.1f}x"


class TestConcurrentCacheAccess:
    """Test cache behavior under concurrent access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_writes(self):
        """Test cache handles concurrent writes correctly."""
        from app.cache.semantic_cache import SemanticCache
        import numpy as np
        
        cache = SemanticCache(
            similarity_threshold=0.95,
            max_size=100,
            ttl_seconds=3600,
        )
        
        async def write_entries(start: int, count: int):
            for i in range(start, start + count):
                vector = np.random.rand(768).tolist()
                cache.set(f"query_{i}", vector, {"data": i})
                await asyncio.sleep(0.001)  # Small delay to simulate real usage
        
        # Launch concurrent writers
        tasks = [
            write_entries(0, 25),
            write_entries(25, 25),
            write_entries(50, 25),
            write_entries(75, 25),
        ]
        
        await asyncio.gather(*tasks)
        
        # Cache should handle concurrent access without corruption
        assert len(cache._cache) == 100, f"Cache should have 100 entries, has {len(cache._cache)}"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_reads(self):
        """Test cache handles concurrent reads correctly."""
        from app.cache.semantic_cache import SemanticCache
        import numpy as np
        
        cache = SemanticCache(
            similarity_threshold=0.95,
            max_size=100,
            ttl_seconds=3600,
        )
        
        # Populate cache
        vectors = []
        for i in range(50):
            vector = np.random.rand(768).tolist()
            vectors.append(vector)
            cache.set(f"query_{i}", vector, {"data": i})
        
        read_results = []
        
        async def read_entries(count: int):
            results = []
            for _ in range(count):
                idx = random.randint(0, 49)
                result = cache.get(f"query_{idx}", vectors[idx])
                results.append(result is not None)
                await asyncio.sleep(0.001)
            return results
        
        # Launch concurrent readers
        tasks = [read_entries(50) for _ in range(10)]
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        total_reads = sum(len(r) for r in all_results)
        total_hits = sum(sum(r) for r in all_results)
        
        print(f"\nConcurrent read test:")
        print(f"  Total reads: {total_reads}")
        print(f"  Total hits: {total_hits}")
        print(f"  Hit rate: {total_hits/total_reads:.2%}")
        
        # Should have very high hit rate for exact matches
        assert total_hits / total_reads > 0.95


class TestHybridSearchPerformance:
    """Performance tests for hybrid search."""
    
    def test_hybrid_search_rrf_speed(self):
        """Test RRF fusion performance."""
        from app.vectorstore.hybrid_search import HybridSearcher, SearchResult
        
        searcher = HybridSearcher(vector_weight=0.7, keyword_weight=0.3)
        
        # Create mock results
        def create_results(count: int, source: str) -> List[SearchResult]:
            return [
                SearchResult(
                    id=f"{source}_{i}",
                    text=f"Document {i} from {source}",
                    score=1.0 - (i / count),
                    payload={"index": i},
                    source=source,
                )
                for i in range(count)
            ]
        
        vector_results = create_results(100, "vector")
        keyword_results = create_results(100, "keyword")
        
        # Measure RRF fusion time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            fused = searcher.reciprocal_rank_fusion(
                [vector_results, keyword_results],
                weights=[0.7, 0.3],
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = statistics.mean(times)
        
        print(f"\nRRF fusion time (100 results each list):")
        print(f"  Average: {avg_time:.3f}ms")
        
        # Should be very fast (<5ms)
        assert avg_time < 5, f"RRF fusion too slow: {avg_time:.3f}ms"
    
    def test_keyword_search_speed(self):
        """Test keyword search performance."""
        from app.vectorstore.hybrid_search import KeywordSearcher
        
        searcher = KeywordSearcher()
        
        # Create mock documents
        documents = [
            {
                "text": f"This is document {i} about meetings and discussions regarding project {i % 10}",
                "title": f"Meeting {i}",
                "date": f"2024-01-{(i % 28) + 1:02d}",
            }
            for i in range(1000)
        ]
        
        query = "What meetings discussed the project timeline?"
        
        # Measure search time
        times = []
        for _ in range(50):
            start = time.perf_counter()
            results = searcher.search(query, documents, limit=10)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = statistics.mean(times)
        
        print(f"\nKeyword search time (1000 documents):")
        print(f"  Average: {avg_time:.2f}ms")
        
        # Should complete in reasonable time (<50ms for 1000 docs)
        assert avg_time < 50, f"Keyword search too slow: {avg_time:.2f}ms"


class TestRateLimiterPerformance:
    """Performance tests for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_throughput(self):
        """Test rate limiter can handle high throughput."""
        from app.middleware.rate_limit import SlidingWindowRateLimiter, RateLimitConfig
        from unittest.mock import MagicMock
        
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=10000,
            requests_per_day=100000,
        )
        
        limiter = SlidingWindowRateLimiter(config)
        
        # Create mock request
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.url.path = "/api/v1/test"
        
        # Measure throughput
        start = time.perf_counter()
        requests = 1000
        
        for _ in range(requests):
            allowed, _ = await limiter.check_rate_limit(mock_request)
        
        elapsed = time.perf_counter() - start
        throughput = requests / elapsed
        
        print(f"\nRate limiter throughput:")
        print(f"  {requests} checks in {elapsed:.3f}s")
        print(f"  Throughput: {throughput:.0f} checks/second")
        
        # Should handle >10k checks/second
        assert throughput > 10000, f"Rate limiter too slow: {throughput:.0f}/s"
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_checks(self):
        """Test rate limiter under concurrent access."""
        from app.middleware.rate_limit import SlidingWindowRateLimiter, RateLimitConfig
        from unittest.mock import MagicMock
        
        config = RateLimitConfig(
            requests_per_minute=100,  # Low limit to test blocking
            requests_per_hour=1000,
            requests_per_day=10000,
        )
        
        limiter = SlidingWindowRateLimiter(config)
        
        async def make_requests(client_id: str, count: int) -> int:
            mock_request = MagicMock()
            mock_request.client.host = client_id
            mock_request.headers = {}
            mock_request.url.path = "/api/v1/test"
            
            allowed_count = 0
            for _ in range(count):
                allowed, _ = await limiter.check_rate_limit(mock_request)
                if allowed:
                    allowed_count += 1
            return allowed_count
        
        # Launch concurrent clients
        tasks = [
            make_requests(f"client_{i}", 50)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        print(f"\nConcurrent rate limiting (10 clients, 50 requests each):")
        for i, count in enumerate(results):
            print(f"  Client {i}: {count}/50 allowed")
        
        # Each client should be limited independently
        # Should allow ~100 per client (limit per minute)
        for count in results:
            assert count == 50, f"Unexpected allowed count: {count}"


class TestEndToEndPerformance:
    """End-to-end performance tests."""
    
    @pytest.mark.asyncio
    async def test_chat_service_with_caching(self):
        """Test ChatService performance with caching enabled."""
        from app.services.chat_service import ChatService
        from unittest.mock import AsyncMock, patch
        import numpy as np
        
        # Mock the dependencies
        mock_chunks = [
            {
                "text": "The meeting discussed Q4 budget allocation of $5M",
                "title": "Budget Meeting",
                "date": "2024-12-15",
                "transcript_id": "t123",
                "score": 0.95,
            }
        ]
        
        mock_retrieval_result = {
            "chunks": mock_chunks,
            "confidence": {"overall": 0.9, "sources": ["Budget Meeting"]},
            "query_analysis": {"intent": "factual", "entities": ["budget"]},
            "disclaimer": None,
        }
        
        with patch.object(ChatService, '__init__', lambda self: None):
            service = ChatService()
            service.retrieval = AsyncMock()
            service.retrieval.search_with_confidence = AsyncMock(return_value=mock_retrieval_result)
            service.retrieval._get_embedding = AsyncMock(return_value=np.random.rand(768).tolist())
            service.llm = AsyncMock()
            service.llm.generate = AsyncMock(return_value="The budget was $5M for Q4.")
            service.prompt_builder = MagicMock()
            service.prompt_builder.build_chat_prompt = MagicMock(return_value="test prompt")
            service.validator = MagicMock()
            service.validator.validate = MagicMock(return_value={"answer": "The budget was $5M for Q4."})
            service.confidence_scorer = MagicMock()
            
            # Initialize cache
            from app.cache.semantic_cache import ResponseCache
            service._response_cache = ResponseCache(
                similarity_threshold=0.92,
                max_size=100,
                ttl_seconds=1800,
            )
            
            # First request (cache miss)
            start = time.perf_counter()
            result1 = await service.answer("What was the Q4 budget?", use_cache=True)
            first_time = (time.perf_counter() - start) * 1000
            
            # Second request (cache hit)
            start = time.perf_counter()
            result2 = await service.answer("What was the Q4 budget?", use_cache=True)
            second_time = (time.perf_counter() - start) * 1000
            
            print(f"\nChatService with caching:")
            print(f"  First request: {first_time:.2f}ms")
            print(f"  Second request: {second_time:.2f}ms")
            print(f"  Cache hit: {result2.get('cache_hit', False)}")


# Pytest configuration
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
