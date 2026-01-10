from fastapi import APIRouter, Query, Body
from qdrant_client.http import models as qm
from typing import Optional, List

from app.core.config import settings
from app.embeddings.client import OllamaEmbeddingClient
from app.vectorstore.qdrant import QdrantStore
from app.services.retrieval_service import RetrievalService
from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    MetadataFilter,
)

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])

store = QdrantStore()
embedder = OllamaEmbeddingClient()
retrieval_service = RetrievalService()  # Enhanced retrieval service
collection = settings.QDRANT_COLLECTION_TRANSCRIPTS


def build_filter(metadata: Optional[MetadataFilter]) -> Optional[qm.Filter]:
    """Build Qdrant filter from metadata filters."""
    if not metadata:
        return None
    
    conditions = []
    
    if metadata.organizer_email:
        conditions.append(
            qm.FieldCondition(
                key="organizer_email",
                match=qm.MatchValue(value=metadata.organizer_email),
            )
        )
    
    if metadata.speakers:
        # Match any of the provided speakers
        conditions.append(
            qm.FieldCondition(
                key="speakers",
                match=qm.MatchAny(any=metadata.speakers),
            )
        )
    
    if metadata.source_file:
        conditions.append(
            qm.FieldCondition(
                key="source_file",
                match=qm.MatchValue(value=metadata.source_file),
            )
        )
    
    if metadata.transcript_id:
        conditions.append(
            qm.FieldCondition(
                key="transcript_id",
                match=qm.MatchValue(value=metadata.transcript_id),
            )
        )
    
    if metadata.date_from or metadata.date_to:
        range_condition = {}
        if metadata.date_from:
            range_condition["gte"] = metadata.date_from
        if metadata.date_to:
            range_condition["lte"] = metadata.date_to
        
        conditions.append(
            qm.FieldCondition(
                key="date",
                range=qm.Range(**range_condition),
            )
        )
    
    if not conditions:
        return None
    
    return qm.Filter(must=conditions)


@router.post("/preview", response_model=RetrievalResponse)
async def preview_retrieval(
    request: RetrievalRequest = Body(...),
):
    """
    Preview retrieval results with metadata filtering.
    Now uses enhanced hybrid search with re-ranking and confidence scoring.
    """
    # Use the enhanced retrieval service
    result = await retrieval_service.search_with_confidence(
        query=request.query,
        limit=request.limit,
        metadata_filter=request.filters,
    )
    
    hits = result["chunks"]
    confidence = result["confidence"]
    query_analysis = result["query_analysis"]
    
    # Format results
    results = []
    for h in hits:
        text = h.get("text", "")
        
        results.append(
            RetrievalResult(
                score=h.get("score", 0),
                transcript_id=h.get("transcript_id"),
                title=h.get("title"),
                date=h.get("date"),
                organizer_email=h.get("organizer_email"),
                speakers=h.get("speakers"),
                source_file=h.get("source_file"),
                chunk_index=h.get("chunk_index"),
                text=text,
                text_preview=text[:200] + "..." if len(text) > 200 else text,
            )
        )
    
    return RetrievalResponse(
        query=request.query,
        limit=request.limit,
        filters_applied=request.filters,
        results_count=len(results),
        results=results,
        confidence=confidence,
        query_analysis=query_analysis,
    )


@router.get("/search")
async def search(
    q: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=20),
    organizer_email: str | None = None,
    use_hybrid: bool = Query(True, description="Enable hybrid search (vector + keyword)"),
    use_reranking: bool = Query(True, description="Enable result re-ranking"),
):
    """
    Enhanced semantic search with hybrid search, re-ranking, and confidence scoring.
    """
    # Build metadata filter if provided
    metadata_filter = None
    if organizer_email:
        metadata_filter = MetadataFilter(organizer_email=organizer_email)
    
    # Use enhanced retrieval service
    result = await retrieval_service.search_with_confidence(
        query=q,
        limit=limit,
        metadata_filter=metadata_filter,
    )
    
    hits = result["chunks"]
    confidence = result["confidence"]
    
    results = []
    for h in hits:
        results.append({
            "score": h.get("score"),
            "transcript_id": h.get("transcript_id"),
            "title": h.get("title"),
            "date": h.get("date"),
            "organizer_email": h.get("organizer_email"),
            "speakers": h.get("speakers", []),
            "chunk_index": h.get("chunk_index"),
            "text": h.get("text"),
            "search_source": h.get("search_source", "vector"),
        })

    return {
        "query": q,
        "limit": limit,
        "results": results,
        "confidence": confidence,
        "query_analysis": result.get("query_analysis"),
    }
