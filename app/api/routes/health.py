from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse
from app.services.health_service import HealthService
from app.services.chat_service import ChatService
from app.core.metrics import metrics

router = APIRouter()

health_service = HealthService()
chat_service = ChatService()


@router.get("/health")
async def health_check():
    return await health_service.full_health()


@router.get("/metrics")
async def get_metrics(
    window: int = Query(default=300, description="Time window in seconds for rolling stats"),
    format: str = Query(default="json", description="Output format: json or prometheus"),
):
    """
    Get comprehensive performance metrics for LLM and RAG pipeline.
    
    Includes:
    - LLM latency (p50, p95, p99), tokens/sec, error rates
    - Retrieval latency breakdown (embedding, vector search, reranking)
    - Cache hit rates
    - End-to-end request metrics
    """
    if format == "prometheus":
        return PlainTextResponse(
            content=metrics.get_prometheus_metrics(),
            media_type="text/plain",
        )
    return {
        "status": "ok",
        "metrics": metrics.get_stats(window_seconds=window),
    }


@router.get("/cache/stats")
async def cache_stats():
    """Get RAG pipeline cache statistics."""
    return {
        "status": "ok",
        "caches": chat_service.get_cache_stats(),
    }


@router.post("/cache/clear")
async def clear_cache():
    """Clear all RAG pipeline caches."""
    chat_service._response_cache.clear()
    chat_service.retrieval._embedding_cache.clear()
    return {
        "status": "ok",
        "message": "All caches cleared",
    }
