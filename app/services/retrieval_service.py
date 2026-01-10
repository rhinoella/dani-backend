from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from hashlib import md5
from qdrant_client.http import models as qm

from app.vectorstore.qdrant import QdrantStore
from app.embeddings.client import OllamaEmbeddingClient
from app.core.config import settings
from app.core.metrics import metrics
from app.schemas.retrieval import MetadataFilter
from app.cache.semantic_cache import SemanticCache
from app.vectorstore.hybrid_search import (
    SearchResult, 
    HybridSearcher, 
    AdaptiveRetriever,
    ReRanker,
    CrossEncoderReRanker,
)
from app.utils.query_processor import QueryProcessor, ConfidenceScorer
from app.utils.meeting_category import matches_meeting_category, infer_meeting_category

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Enhanced retrieval service with:
    - Semantic caching for embeddings
    - Hybrid search (vector + keyword)
    - Adaptive top-k retrieval
    - Multi-stage re-ranking (heuristic + cross-encoder)
    - Confidence scoring
    - Query expansion
    """

    def __init__(self) -> None:
        self.store = QdrantStore()
        self.embedder = OllamaEmbeddingClient()
        
        # Semantic cache for embeddings (similarity-based)
        self._embedding_cache = SemanticCache(
            similarity_threshold=0.98,  # High threshold for embeddings
            max_size=500,
            ttl_seconds=3600,  # 1 hour
        )
        
        # RAG optimization components with improved settings
        self.hybrid_searcher = HybridSearcher(
            vector_weight=settings.HYBRID_VECTOR_WEIGHT,
            keyword_weight=settings.HYBRID_KEYWORD_WEIGHT,
        )
        self.adaptive_retriever = AdaptiveRetriever(
            min_similarity=settings.ADAPTIVE_MIN_SIMILARITY,
            max_chunks=settings.ADAPTIVE_MAX_CHUNKS,
            min_chunks=settings.ADAPTIVE_MIN_CHUNKS,
        )
        self.reranker = ReRanker()
        self.cross_encoder_reranker = CrossEncoderReRanker(top_k=10)
        self.query_processor = QueryProcessor()
        self.confidence_scorer = ConfidenceScorer()

    def _cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return md5(query.lower().strip().encode()).hexdigest()
    
    async def _get_embedding(self, query: str) -> List[float]:
        """Get query embedding with semantic caching and proper prefix."""
        # Try semantic cache first
        cached = self._embedding_cache.get(query)
        if cached is not None:
            return cached[0]  # Return cached vector
        
        # Generate new embedding with search_query prefix for better relevance
        query_vector = await self.embedder.embed_query(query)
        
        # Cache with the vector itself as the key vector
        self._embedding_cache.set(query, query_vector, query_vector)
        
        return query_vector
    
    def _build_filter(self, metadata: Optional[MetadataFilter]) -> Optional[qm.Filter]:
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
        
    # Document type filter
        if metadata.doc_type and metadata.doc_type != "all":
            conditions.append(
                qm.FieldCondition(
                    key="doc_type",
                    match=qm.MatchValue(value=metadata.doc_type),
                )
            )

        if metadata.document_ids:
            # Match any of the provided document IDs
            # Note: In document collection it's 'document_id', in transcripts it might be absent
            # For now we assume we are searching documents collection if explicit document_ids are provided
            # effectively boosting them or filtering for them.
            conditions.append(
                qm.FieldCondition(
                    key="document_id",
                    match=qm.MatchAny(any=metadata.document_ids),
                )
            )
        
        if not conditions:
            return None
        
        return qm.Filter(must=conditions)
    
    def _build_document_filter(self, metadata: Optional[MetadataFilter]) -> Optional[qm.Filter]:
        """Build Qdrant filter specifically for the documents collection."""
        if not metadata:
            return None
            
        conditions = []
        
        # document_ids are the primary filter for documents
        if metadata.document_ids:
            # In documents collection, 'document_id' is the primary key
            conditions.append(
                qm.FieldCondition(
                    key="document_id",
                    match=qm.MatchAny(any=metadata.document_ids),
                )
            )
            
        # Documents can also have chunk_index, etc, but typically we filter docs by ID
        
        if not conditions:
            return None
            
        return qm.Filter(must=conditions)
    
    def _apply_category_filter(
        self,
        results: List[SearchResult],
        category: str,
    ) -> List[SearchResult]:
        """
        Apply meeting category filter to search results using runtime inference.
        
        This enables filtering by meeting type (board, 1on1, standup, etc.)
        without requiring changes to the ingestion schema.
        
        Args:
            results: List of search results to filter
            category: Meeting category to filter by
            
        Returns:
            Filtered list of results matching the category
        """
        if not category or category == "all":
            return results
        
        filtered = []
        for result in results:
            title = result.payload.get("title")
            organizer_email = result.payload.get("organizer_email")
            speakers = result.payload.get("speakers", [])
            
            if matches_meeting_category(
                target_category=category,
                title=title,
                organizer_email=organizer_email,
                speakers=speakers,
            ):
                filtered.append(result)
        
        logger.debug(
            f"Category filter '{category}': {len(results)} -> {len(filtered)} results"
        )
        return filtered

    async def search(
        self, 
        query: str, 
        limit: int = 6,
        metadata_filter: Optional[MetadataFilter] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_adaptive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with hybrid search, re-ranking, and adaptive retrieval.
        
        Args:
            query: Search query
            limit: Maximum results to return
            metadata_filter: Optional metadata filters
            use_hybrid: Enable hybrid (vector + keyword) search
            use_reranking: Enable result re-ranking
            use_adaptive: Enable adaptive top-k retrieval
        
        Returns:
            List of relevant chunks with metadata
        """
        import time
        timings = {}
        
        # Process query for better retrieval
        intent_start = time.time()
        intent = self.query_processor.detect_intent(query)
        timings["intent_ms"] = round((time.time() - intent_start) * 1000, 2)
        logger.debug(f"Query intent: {intent.intent_type}, entities: {intent.entities}")
        
        # Get embedding (with caching and search_query prefix)
        embed_start = time.time()
        query_vector = await self._get_embedding(query)
        timings["embedding_ms"] = round((time.time() - embed_start) * 1000, 2)
        
        # Build filter
        filter_ = self._build_filter(metadata_filter)

        # Fetch more results initially for hybrid search / reranking (4x for better candidate pool)
        initial_limit = limit * 4 if (use_hybrid or use_reranking) else limit

        # Vector search - search transcripts collection
        vector_start = time.time()
        hits = await self.store.search(
            collection=settings.QDRANT_COLLECTION_TRANSCRIPTS,
            query_vector=query_vector,
            limit=initial_limit,
            filter_=filter_,
        )
        timings["vector_search_ms"] = round((time.time() - vector_start) * 1000, 2)
        
        # Also search documents collection if it exists
        doc_hits = []
        try:
            doc_start = time.time()
            doc_hits = await self.store.search(
                collection=settings.QDRANT_COLLECTION_DOCUMENTS,
                query_vector=query_vector,
                limit=initial_limit,
                filter_=self._build_document_filter(metadata_filter),  # Use specialized filter for documents
            )
            timings["document_search_ms"] = round((time.time() - doc_start) * 1000, 2)
            logger.debug(f"Found {len(doc_hits)} results from documents collection")
        except Exception as e:
            # Documents collection may not exist or be empty
            logger.debug(f"Documents collection search skipped: {e}")

        # Convert to SearchResult objects
        vector_results: List[SearchResult] = []
        all_documents: List[Dict[str, Any]] = []
        
        # Process transcript hits
        for h in hits:
            payload = h.payload or {}
            doc = {
                "id": str(h.id),
                "text": payload.get("text", ""),
                "score": h.score,
                "title": payload.get("title"),
                "date": payload.get("date"),
                "transcript_id": payload.get("transcript_id"),
                "organizer_email": payload.get("organizer"),
                "chunk_index": payload.get("chunk_index"),
                "speakers": payload.get("speakers", []),
                "source_file": payload.get("source_file"),
            }
            
            vector_results.append(SearchResult(
                id=str(h.id),
                text=payload.get("text", ""),
                score=h.score,
                payload={k: v for k, v in doc.items() if k not in ["id", "text", "score"]},
                source="vector",
            ))
            all_documents.append(doc)
        
        # Process document hits (different payload structure)
        for h in doc_hits:
            payload = h.payload or {}
            # Documents store title/filename differently
            doc_title = payload.get("title") or payload.get("filename", "Untitled Document")
            doc = {
                "id": str(h.id),
                "text": payload.get("text", ""),
                "score": h.score,
                "title": doc_title,
                "date": None,  # Documents don't have meeting dates
                "transcript_id": payload.get("document_id"),  # Use document_id as transcript_id
                "organizer_email": None,
                "chunk_index": payload.get("chunk_index"),
                "speakers": [],  # Documents don't have speakers
                "source_file": f"document:{payload.get('document_id', '')}",
                "document_source": True,  # Flag to indicate this is from a document
            }
            
            vector_results.append(SearchResult(
                id=str(h.id),
                text=payload.get("text", ""),
                score=h.score,
                payload={k: v for k, v in doc.items() if k not in ["id", "text", "score"]},
                source="document",
            ))
            all_documents.append(doc)

        
        # Apply hybrid search if enabled
        hybrid_start = time.time()
        if use_hybrid and all_documents:
            results = self.hybrid_searcher.search(
                query=query,
                vector_results=vector_results,
                all_documents=all_documents,
                limit=initial_limit,
            )
        else:
            results = vector_results
        timings["hybrid_search_ms"] = round((time.time() - hybrid_start) * 1000, 2)
        
        # Apply heuristic re-ranking if enabled
        rerank_start = time.time()
        if use_reranking and results:
            results = self.reranker.rerank(
                query=query,
                results=results,
                boost_speakers=intent.entities,  # Boost if query mentions specific people
            )
        timings["heuristic_rerank_ms"] = round((time.time() - rerank_start) * 1000, 2)
        
        # Apply cross-encoder re-ranking for highest quality (on top results only)
        # Uses async version to avoid blocking event loop on CPU
        cross_encoder_start = time.time()
        if use_reranking and results and getattr(settings, 'CROSS_ENCODER_ENABLED', False):
            results = await self.cross_encoder_reranker.rerank_async(query=query, results=results)
            timings["cross_encoder_ms"] = round((time.time() - cross_encoder_start) * 1000, 2)
        
        # Apply adaptive retrieval if enabled
        if use_adaptive and results:
            results, retrieval_meta = self.adaptive_retriever.filter_results(results)
            logger.debug(f"Adaptive retrieval: {retrieval_meta}")
        
        # Apply meeting category filter if specified (runtime inference from title)
        category_filter_start = time.time()
        if metadata_filter and metadata_filter.meeting_category:
            results = self._apply_category_filter(results, metadata_filter.meeting_category)
            timings["category_filter_ms"] = round((time.time() - category_filter_start) * 1000, 2)
        
        # Log retrieval timing breakdown
        total_retrieval = sum(timings.values())
        logger.info(f"Retrieval timing: {timings} (total: {total_retrieval:.0f}ms)")
        
        # Record detailed retrieval metrics
        metrics.record_retrieval(
            latency_ms=total_retrieval,
            chunks_found=len(results),
            embedding_ms=timings.get("embedding_ms", 0),
            vector_search_ms=timings.get("vector_search_ms", 0),
            rerank_ms=timings.get("heuristic_rerank_ms", 0) + timings.get("cross_encoder_ms", 0),
        )
        
        # Limit final results
        results = results[:limit]

        # Convert to output format with inferred meeting category
        output: List[Dict[str, Any]] = []
        for r in results:
            # Infer meeting category from metadata
            title = r.payload.get("title")
            organizer_email = r.payload.get("organizer_email")
            speakers = r.payload.get("speakers", [])
            inferred_category, category_confidence = infer_meeting_category(
                title=title,
                organizer_email=organizer_email,
                speakers=speakers,
            )
            
            output.append({
                "score": r.score,
                "text": r.text,
                "title": title,
                "date": r.payload.get("date"),
                "transcript_id": r.payload.get("transcript_id"),
                "organizer_email": organizer_email,
                "chunk_index": r.payload.get("chunk_index"),
                "speakers": speakers,
                "source_file": r.payload.get("source_file"),
                "search_source": r.source,  # "vector", "keyword", or "hybrid"
                "meeting_category": inferred_category,
                "category_confidence": category_confidence,
            })

        return output
    
    async def search_with_confidence(
        self,
        query: str,
        limit: int = 6,
        metadata_filter: Optional[MetadataFilter] = None,
    ) -> Dict[str, Any]:
        """
        Search with confidence scoring.
        
        Returns:
            Dict with chunks, confidence metrics, and query analysis
        """
        intent = self.query_processor.detect_intent(query)
        chunks = await self.search(query, limit, metadata_filter)
        confidence = self.confidence_scorer.score(chunks, query)
        
        return {
            "chunks": chunks,
            "confidence": confidence,
            "query_analysis": {
                "intent": intent.intent_type,
                "entities": intent.entities,
                "time_references": intent.time_references,
                "processed_query": intent.processed_query,
            },
            "disclaimer": self.confidence_scorer.get_disclaimer(confidence),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self._embedding_cache.get_stats()
