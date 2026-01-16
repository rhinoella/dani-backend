"""
Semantic caching for RAG pipeline.

Provides:
- SemanticCache: Cache based on query semantic similarity
- ResponseCache: Full LLM response caching with TTL
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from hashlib import md5

from app.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    query: str
    query_vector: List[float]
    response: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > ttl_seconds


class SemanticCache:
    """
    Semantic similarity-based cache for embeddings and responses.
    
    Instead of exact match, finds cached entries with similar semantic meaning.
    Uses cosine similarity to compare query embeddings.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_size: int = 500,
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # OrderedDict for LRU eviction
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0  # Hits via similarity (not exact match)
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key from query text."""
        return md5(query.lower().strip().encode()).hexdigest()
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self.max_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired(self.ttl_seconds)
        ]
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get(
        self, 
        query: str, 
        query_vector: Optional[List[float]] = None
    ) -> Optional[Tuple[Any, float]]:
        """
        Get cached response for query.
        
        Args:
            query: Query text
            query_vector: Query embedding (optional, for semantic matching)
        
        Returns:
            Tuple of (cached_response, similarity_score) or None if not found
        """
        # Cleanup expired entries periodically
        if len(self._cache) > 0 and self.hits + self.misses > 0 and (self.hits + self.misses) % 100 == 0:
            self._cleanup_expired()
        
        # Try exact match first
        key = self._generate_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired(self.ttl_seconds):
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.hits += 1
                self.hits += 1
                logger.debug(f"Cache HIT (exact): {query[:50]}...")
                return (entry.response, 1.0)
        
        # Try semantic matching if vector provided
        if query_vector is not None:
            best_match: Optional[CacheEntry] = None
            best_similarity = 0.0
            
            for entry in self._cache.values():
                if entry.is_expired(self.ttl_seconds):
                    continue
                
                similarity = cosine_similarity(query_vector, entry.query_vector)
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
            
            if best_match is not None:
                best_match.hits += 1
                self.hits += 1
                self.semantic_hits += 1
                logger.debug(f"Cache HIT (semantic, sim={best_similarity:.3f}): {query[:50]}...")
                return (best_match.response, best_similarity)
        
        self.misses += 1
        logger.debug(f"Cache MISS: {query[:50]}...")
        return None
    
    def set(
        self,
        query: str,
        query_vector: List[float],
        response: Any,
    ) -> None:
        """
        Cache a response for a query.
        
        Args:
            query: Query text
            query_vector: Query embedding
            response: Response to cache
        """
        self._evict_if_needed()
        
        key = self._generate_key(query)
        entry = CacheEntry(
            query=query,
            query_vector=query_vector,
            response=response,
        )
        
        self._cache[key] = entry
        self._cache.move_to_end(key)
        
        logger.debug(f"Cache SET: {query[:50]}...")
    
    def invalidate(self, query: str) -> bool:
        """Invalidate a specific cache entry."""
        key = self._generate_key(query)
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self.ttl_seconds,
        }
    
    # Alias for backwards compatibility
    stats = get_stats


class ResponseCache:
    """
    Full LLM response caching with stampede protection.
    
    Caches complete responses including:
    - Answer text
    - Sources
    - Confidence scores
    
    Uses both exact and semantic matching.
    Includes lock-based stampede protection to prevent multiple
    concurrent requests from triggering expensive LLM calls.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,  # Slightly lower for full responses
        max_size: int = 200,
        ttl_seconds: int = 1800,  # 30 min default for full responses
    ):
        self._semantic_cache = SemanticCache(
            similarity_threshold=similarity_threshold,
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )
        # Stampede protection: locks per cache key
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects the locks dict
        self._lock_access_count = 0  # Track accesses for periodic cleanup
    
    async def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a cache key."""
        async with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            
            # Periodic cleanup every 100 accesses
            self._lock_access_count += 1
            if self._lock_access_count % 100 == 0:
                await self._cleanup_stale_locks()
            
            return self._locks[key]
    
    async def _cleanup_lock(self, key: str) -> None:
        """Remove lock if no longer needed (no waiters)."""
        async with self._locks_lock:
            lock = self._locks.get(key)
            if lock and not lock.locked():
                del self._locks[key]
    
    async def _cleanup_stale_locks(self) -> None:
        """Remove all unlocked locks to prevent memory leak."""
        stale_keys = [key for key, lock in self._locks.items() if not lock.locked()]
        for key in stale_keys:
            del self._locks[key]
        
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale locks")
    
    async def get_or_generate(
        self,
        query: str,
        query_vector: List[float],
        generator_fn,  # async function that generates response
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Get cached response or generate new one with stampede protection.
        
        Args:
            query: User query
            query_vector: Query embedding
            generator_fn: Async function to generate response if cache miss
        
        Returns:
            Tuple of (response_dict, was_cached)
        """
        # Generate cache key for lock
        cache_key = md5(query.lower().strip().encode()).hexdigest()
        
        # Try cache first (outside lock for fast path)
        cached = self._semantic_cache.get(query, query_vector)
        if cached is not None:
            response, similarity = cached
            response["_cache"] = {
                "hit": True,
                "similarity": round(similarity, 3),
            }
            return (response, True)
        
        # Cache miss - acquire lock to prevent stampede
        lock = await self._get_lock(cache_key)
        
        async with lock:
            # Double-check cache after acquiring lock (another request may have populated it)
            cached = self._semantic_cache.get(query, query_vector)
            if cached is not None:
                response, similarity = cached
                response["_cache"] = {
                    "hit": True,
                    "similarity": round(similarity, 3),
                    "stampede_protected": True,
                }
                return (response, True)
            
            # Generate new response (only one request does this)
            response = await generator_fn()
            
            # Cache the response
            self._semantic_cache.set(query, query_vector, response)
            
            response["_cache"] = {"hit": False}
        
        # Cleanup lock if no longer needed
        await self._cleanup_lock(cache_key)
        
        return (response, False)
    
    def invalidate(self, query: str) -> bool:
        """Invalidate cached response for query."""
        return self._semantic_cache.invalidate(query)
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self._semantic_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._semantic_cache.get_stats()
