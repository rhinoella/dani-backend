"""
Performance metrics collection for LLM and RAG monitoring.

Provides:
- Request latency tracking (p50, p95, p99)
- Throughput metrics (requests/sec, tokens/sec)
- Error rate tracking
- Cache hit rates
- Rolling window statistics

Usage:
    from app.core.metrics import metrics
    
    # Record a metric
    metrics.record_llm_request(latency_ms=5000, tokens_in=500, tokens_out=200)
    metrics.record_retrieval(latency_ms=150, chunks_found=5, cache_hit=False)
    
    # Get stats
    stats = metrics.get_stats()
"""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Deque
from threading import Lock
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricWindow:
    """Rolling window for metric collection."""
    
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _lock: Lock = field(default_factory=Lock)
    
    def add(self, value: float) -> None:
        """Add a value to the window."""
        with self._lock:
            self.values.append(value)
            self.timestamps.append(time.time())
    
    def get_stats(self, window_seconds: int = 300) -> Dict[str, float]:
        """Get statistics for values within the time window."""
        with self._lock:
            now = time.time()
            cutoff = now - window_seconds
            
            # Filter to recent values
            recent = [v for v, t in zip(self.values, self.timestamps) if t > cutoff]
            
            if not recent:
                return {
                    "count": 0,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0,
                }
            
            sorted_values = sorted(recent)
            n = len(sorted_values)
            
            return {
                "count": n,
                "mean": round(statistics.mean(recent), 2),
                "min": round(min(recent), 2),
                "max": round(max(recent), 2),
                "p50": round(sorted_values[int(n * 0.5)] if n > 0 else 0, 2),
                "p95": round(sorted_values[int(n * 0.95)] if n > 1 else sorted_values[-1], 2),
                "p99": round(sorted_values[int(n * 0.99)] if n > 1 else sorted_values[-1], 2),
            }
    
    def get_rate(self, window_seconds: int = 60) -> float:
        """Get requests per second over the window."""
        with self._lock:
            now = time.time()
            cutoff = now - window_seconds
            count = sum(1 for t in self.timestamps if t > cutoff)
            return round(count / window_seconds, 3)


@dataclass
class CounterMetric:
    """Simple counter metric."""
    
    value: int = 0
    _lock: Lock = field(default_factory=Lock)
    
    def increment(self, by: int = 1) -> None:
        with self._lock:
            self.value += by
    
    def get(self) -> int:
        with self._lock:
            return self.value


class MetricsCollector:
    """
    Centralized metrics collection for the RAG pipeline.
    
    Tracks:
    - LLM performance (latency, tokens, throughput)
    - Retrieval performance (embedding, search, reranking)
    - Cache performance (hit rates)
    - Error rates
    """
    
    def __init__(self):
        # LLM Metrics
        self.llm_latency = MetricWindow()
        self.llm_tokens_in = MetricWindow()
        self.llm_tokens_out = MetricWindow()
        self.llm_tokens_per_sec = MetricWindow()
        self.llm_requests = CounterMetric()
        self.llm_errors = CounterMetric()
        
        # Retrieval Metrics
        self.retrieval_latency = MetricWindow()
        self.embedding_latency = MetricWindow()
        self.vector_search_latency = MetricWindow()
        self.rerank_latency = MetricWindow()
        self.chunks_retrieved = MetricWindow()
        self.retrieval_requests = CounterMetric()
        
        # Cache Metrics
        self.cache_hits = CounterMetric()
        self.cache_misses = CounterMetric()
        self.embedding_cache_hits = CounterMetric()
        self.embedding_cache_misses = CounterMetric()
        
        # End-to-end Metrics
        self.e2e_latency = MetricWindow()
        self.total_requests = CounterMetric()
        self.total_errors = CounterMetric()
        
        # Startup time
        self._start_time = time.time()
    
    def record_llm_request(
        self,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: bool = False,
    ) -> None:
        """Record an LLM request."""
        self.llm_latency.add(latency_ms)
        self.llm_requests.increment()
        
        if tokens_in > 0:
            self.llm_tokens_in.add(tokens_in)
        if tokens_out > 0:
            self.llm_tokens_out.add(tokens_out)
            # Calculate tokens per second
            if latency_ms > 0:
                tps = (tokens_out / latency_ms) * 1000
                self.llm_tokens_per_sec.add(tps)
        
        if error:
            self.llm_errors.increment()
            self.total_errors.increment()
        
        logger.debug(f"LLM request: {latency_ms:.0f}ms, {tokens_in} in, {tokens_out} out")
    
    def record_retrieval(
        self,
        latency_ms: float,
        chunks_found: int = 0,
        embedding_ms: float = 0,
        vector_search_ms: float = 0,
        rerank_ms: float = 0,
        cache_hit: bool = False,
    ) -> None:
        """Record a retrieval request with breakdown."""
        self.retrieval_latency.add(latency_ms)
        self.retrieval_requests.increment()
        
        if chunks_found > 0:
            self.chunks_retrieved.add(chunks_found)
        if embedding_ms > 0:
            self.embedding_latency.add(embedding_ms)
        if vector_search_ms > 0:
            self.vector_search_latency.add(vector_search_ms)
        if rerank_ms > 0:
            self.rerank_latency.add(rerank_ms)
        
        logger.debug(f"Retrieval: {latency_ms:.0f}ms, {chunks_found} chunks")
    
    def record_cache_access(self, hit: bool, cache_type: str = "response") -> None:
        """Record a cache access."""
        if cache_type == "response":
            if hit:
                self.cache_hits.increment()
            else:
                self.cache_misses.increment()
        elif cache_type == "embedding":
            if hit:
                self.embedding_cache_hits.increment()
            else:
                self.embedding_cache_misses.increment()
    
    def record_e2e_request(self, latency_ms: float, error: bool = False) -> None:
        """Record an end-to-end request."""
        self.e2e_latency.add(latency_ms)
        self.total_requests.increment()
        if error:
            self.total_errors.increment()
    
    def get_stats(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Get comprehensive metrics stats."""
        
        # Calculate cache hit rates
        response_cache_total = self.cache_hits.get() + self.cache_misses.get()
        response_cache_hit_rate = (
            self.cache_hits.get() / response_cache_total 
            if response_cache_total > 0 else 0
        )
        
        embedding_cache_total = self.embedding_cache_hits.get() + self.embedding_cache_misses.get()
        embedding_cache_hit_rate = (
            self.embedding_cache_hits.get() / embedding_cache_total
            if embedding_cache_total > 0 else 0
        )
        
        # Calculate error rate
        total = self.total_requests.get()
        error_rate = self.total_errors.get() / total if total > 0 else 0
        
        return {
            "uptime_seconds": round(time.time() - self._start_time, 0),
            "window_seconds": window_seconds,
            
            # LLM Metrics
            "llm": {
                "latency_ms": self.llm_latency.get_stats(window_seconds),
                "tokens_in": self.llm_tokens_in.get_stats(window_seconds),
                "tokens_out": self.llm_tokens_out.get_stats(window_seconds),
                "tokens_per_second": self.llm_tokens_per_sec.get_stats(window_seconds),
                "requests_total": self.llm_requests.get(),
                "requests_per_second": self.llm_latency.get_rate(60),
                "errors_total": self.llm_errors.get(),
            },
            
            # Retrieval Metrics
            "retrieval": {
                "latency_ms": self.retrieval_latency.get_stats(window_seconds),
                "embedding_latency_ms": self.embedding_latency.get_stats(window_seconds),
                "vector_search_latency_ms": self.vector_search_latency.get_stats(window_seconds),
                "rerank_latency_ms": self.rerank_latency.get_stats(window_seconds),
                "chunks_retrieved": self.chunks_retrieved.get_stats(window_seconds),
                "requests_total": self.retrieval_requests.get(),
                "requests_per_second": self.retrieval_latency.get_rate(60),
            },
            
            # Cache Metrics
            "cache": {
                "response_cache": {
                    "hits": self.cache_hits.get(),
                    "misses": self.cache_misses.get(),
                    "hit_rate": round(response_cache_hit_rate, 3),
                },
                "embedding_cache": {
                    "hits": self.embedding_cache_hits.get(),
                    "misses": self.embedding_cache_misses.get(),
                    "hit_rate": round(embedding_cache_hit_rate, 3),
                },
            },
            
            # End-to-end Metrics
            "e2e": {
                "latency_ms": self.e2e_latency.get_stats(window_seconds),
                "requests_total": self.total_requests.get(),
                "requests_per_second": self.e2e_latency.get_rate(60),
                "errors_total": self.total_errors.get(),
                "error_rate": round(error_rate, 4),
            },
        }
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        stats = self.get_stats()
        lines = []
        
        # Helper to add metric
        def add_metric(name: str, value: float, help_text: str, metric_type: str = "gauge"):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {value}")
        
        # LLM metrics
        add_metric("dani_llm_latency_p50_ms", stats["llm"]["latency_ms"]["p50"], "LLM latency p50 in ms")
        add_metric("dani_llm_latency_p95_ms", stats["llm"]["latency_ms"]["p95"], "LLM latency p95 in ms")
        add_metric("dani_llm_latency_p99_ms", stats["llm"]["latency_ms"]["p99"], "LLM latency p99 in ms")
        add_metric("dani_llm_tokens_per_second", stats["llm"]["tokens_per_second"]["mean"], "LLM tokens per second")
        add_metric("dani_llm_requests_total", stats["llm"]["requests_total"], "Total LLM requests", "counter")
        add_metric("dani_llm_errors_total", stats["llm"]["errors_total"], "Total LLM errors", "counter")
        
        # Retrieval metrics
        add_metric("dani_retrieval_latency_p50_ms", stats["retrieval"]["latency_ms"]["p50"], "Retrieval latency p50 in ms")
        add_metric("dani_retrieval_latency_p95_ms", stats["retrieval"]["latency_ms"]["p95"], "Retrieval latency p95 in ms")
        add_metric("dani_retrieval_embedding_latency_p50_ms", stats["retrieval"]["embedding_latency_ms"]["p50"], "Embedding latency p50 in ms")
        add_metric("dani_retrieval_vector_search_latency_p50_ms", stats["retrieval"]["vector_search_latency_ms"]["p50"], "Vector search latency p50 in ms")
        
        # Cache metrics
        add_metric("dani_cache_response_hit_rate", stats["cache"]["response_cache"]["hit_rate"], "Response cache hit rate")
        add_metric("dani_cache_embedding_hit_rate", stats["cache"]["embedding_cache"]["hit_rate"], "Embedding cache hit rate")
        
        # E2E metrics
        add_metric("dani_e2e_latency_p50_ms", stats["e2e"]["latency_ms"]["p50"], "End-to-end latency p50 in ms")
        add_metric("dani_e2e_latency_p95_ms", stats["e2e"]["latency_ms"]["p95"], "End-to-end latency p95 in ms")
        add_metric("dani_e2e_requests_total", stats["e2e"]["requests_total"], "Total requests", "counter")
        add_metric("dani_e2e_error_rate", stats["e2e"]["error_rate"], "Error rate")
        add_metric("dani_uptime_seconds", stats["uptime_seconds"], "Uptime in seconds", "counter")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self.__init__()


# Global metrics instance
metrics = MetricsCollector()
