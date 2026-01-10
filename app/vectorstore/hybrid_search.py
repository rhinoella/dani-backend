"""
Hybrid search combining semantic (vector) and lexical (keyword) search.

Implements:
- Reciprocal Rank Fusion (RRF) for merging results
- Adaptive top-k retrieval
- Re-ranking with relevance scoring
- Cross-encoder re-ranking (optional, for production quality)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from collections import Counter

from app.core.config import settings

logger = logging.getLogger(__name__)

# Lazy load cross-encoder to avoid import overhead when not used
_cross_encoder = None
_cross_encoder_loading = False


def get_cross_encoder():
    """Lazy load cross-encoder model."""
    global _cross_encoder, _cross_encoder_loading
    
    if _cross_encoder is not None:
        return _cross_encoder
    
    if _cross_encoder_loading:
        return None  # Prevent recursive loading
    
    if not getattr(settings, 'CROSS_ENCODER_ENABLED', False):
        return None
    
    try:
        _cross_encoder_loading = True
        from sentence_transformers import CrossEncoder
        model_name = getattr(settings, 'CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info(f"Loading cross-encoder model: {model_name}")
        _cross_encoder = CrossEncoder(model_name, max_length=512)
        logger.info("Cross-encoder model loaded successfully")
        return _cross_encoder
    except ImportError:
        logger.warning("sentence-transformers not installed, cross-encoder reranking disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to load cross-encoder model: {e}")
        return None
    finally:
        _cross_encoder_loading = False


@dataclass
class SearchResult:
    """Unified search result."""
    id: str
    text: str
    score: float
    payload: Dict[str, Any]
    source: str = "vector"  # "vector", "keyword", or "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            **self.payload,
        }


class KeywordSearcher:
    """
    Simple keyword-based search using TF-IDF-like scoring.
    Works on chunk text directly without external dependencies.
    """
    
    def __init__(self):
        self._stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "it", "its", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "what", "which", "who", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there",
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text."""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        # Remove stop words and short tokens
        return [w for w in words if w not in self._stop_words and len(w) > 2]
    
    def _compute_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute keyword match score."""
        if not query_tokens or not doc_tokens:
            return 0.0
        
        query_set = set(query_tokens)
        doc_counter = Counter(doc_tokens)
        
        # Calculate match score based on:
        # 1. Term frequency in document
        # 2. Percentage of query terms found
        matches = 0
        weighted_matches = 0.0
        
        for token in query_set:
            if token in doc_counter:
                matches += 1
                # Log-scaled term frequency
                weighted_matches += 1 + (0.5 * min(doc_counter[token], 5))
        
        if matches == 0:
            return 0.0
        
        # Combine coverage and frequency
        coverage = matches / len(query_set)
        frequency_bonus = weighted_matches / len(query_set)
        
        return (coverage * 0.6 + frequency_bonus * 0.4)
    
    def search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        limit: int = 10,
        text_field: str = "text",
    ) -> List[SearchResult]:
        """
        Search documents using keyword matching.
        
        Args:
            query: Search query
            documents: List of documents with text and metadata
            limit: Maximum results to return
            text_field: Field name containing searchable text
        
        Returns:
            List of SearchResult ordered by score
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        results = []
        
        for i, doc in enumerate(documents):
            text = doc.get(text_field, "")
            doc_tokens = self._tokenize(text)
            
            score = self._compute_score(query_tokens, doc_tokens)
            
            if score > 0.1:  # Minimum threshold
                results.append(SearchResult(
                    id=doc.get("id", str(i)),
                    text=text,
                    score=score,
                    payload={k: v for k, v in doc.items() if k != text_field},
                    source="keyword",
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:limit]


class HybridSearcher:
    """
    Combines vector search with keyword search using Reciprocal Rank Fusion.
    
    Optimized for meeting transcripts with proper nouns, names, and specific terms.
    """
    
    def __init__(
        self,
        vector_weight: float = 0.6,   # Reduced from 0.7 - keywords matter for names
        keyword_weight: float = 0.4,  # Increased from 0.3
        rrf_k: int = 40,  # Reduced from 60 for better ranking with smaller result sets
    ):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.keyword_searcher = KeywordSearcher()
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None,
    ) -> List[SearchResult]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(weight / (k + rank)) for each list
        
        Args:
            result_lists: List of ranked result lists
            weights: Optional weights for each list
        
        Returns:
            Merged and re-ranked results
        """
        if not result_lists:
            return []
        
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        
        for list_idx, results in enumerate(result_lists):
            for rank, result in enumerate(results):
                # RRF formula: weight / (k + rank)
                rrf_score = weights[list_idx] / (self.rrf_k + rank + 1)
                
                if result.id in rrf_scores:
                    rrf_scores[result.id] += rrf_score
                else:
                    rrf_scores[result.id] = rrf_score
                    result_map[result.id] = result
        
        # Create merged results with RRF scores
        merged = []
        for result_id, rrf_score in rrf_scores.items():
            result = result_map[result_id]
            merged.append(SearchResult(
                id=result.id,
                text=result.text,
                score=rrf_score,
                payload=result.payload,
                source="hybrid",
            ))
        
        # Sort by RRF score descending
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged
    
    def search(
        self,
        query: str,
        vector_results: List[SearchResult],
        all_documents: List[Dict[str, Any]],
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword results.
        
        Args:
            query: Search query
            vector_results: Results from vector search
            all_documents: All documents for keyword search
            limit: Maximum results to return
        
        Returns:
            Hybrid search results
        """
        # Get keyword results
        keyword_results = self.keyword_searcher.search(
            query=query,
            documents=all_documents,
            limit=limit * 2,  # Get more for merging
        )
        
        # Merge using RRF
        merged = self.reciprocal_rank_fusion(
            [vector_results, keyword_results],
            [self.vector_weight, self.keyword_weight],
        )
        
        return merged[:limit]


class AdaptiveRetriever:
    """
    Adaptive top-k retrieval that adjusts based on relevance score drop-off.
    
    Calibrated for raw cosine similarity which is typically 0.05-0.30 
    for conversational content like meeting transcripts.
    """
    
    def __init__(
        self,
        min_similarity: float = 0.05,   # Lowered from 0.65 - raw cosine is low
        max_chunks: int = 20,           # Increased from 15
        min_chunks: int = 3,
        drop_off_threshold: float = 0.05,  # Lowered from 0.15 - score differences are smaller
    ):
        self.min_similarity = min_similarity
        self.max_chunks = max_chunks
        self.min_chunks = min_chunks
        self.drop_off_threshold = drop_off_threshold
    
    def filter_results(
        self,
        results: List[SearchResult],
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Filter results based on adaptive criteria.
        
        Returns:
            Tuple of (filtered_results, metadata)
        """
        if not results:
            return [], {"reason": "no_results"}
        
        filtered = []
        prev_score = results[0].score if results else 0
        
        for i, result in enumerate(results):
            # Always include minimum chunks
            if i < self.min_chunks:
                filtered.append(result)
                prev_score = result.score
                continue
            
            # Stop if max reached
            if i >= self.max_chunks:
                break
            
            # Stop if below minimum similarity
            if result.score < self.min_similarity:
                break
            
            # Stop if large drop-off from previous
            score_drop = prev_score - result.score
            if score_drop > self.drop_off_threshold:
                break
            
            filtered.append(result)
            prev_score = result.score
        
        metadata = {
            "total_candidates": len(results),
            "filtered_count": len(filtered),
            "top_score": results[0].score if results else 0,
            "bottom_score": filtered[-1].score if filtered else 0,
            "cutoff_reason": self._get_cutoff_reason(results, filtered),
        }
        
        return filtered, metadata
    
    def _get_cutoff_reason(
        self,
        original: List[SearchResult],
        filtered: List[SearchResult],
    ) -> str:
        """Determine why results were cut off."""
        if len(filtered) >= self.max_chunks:
            return "max_chunks_reached"
        if len(filtered) < len(original):
            next_idx = len(filtered)
            if next_idx < len(original):
                if original[next_idx].score < self.min_similarity:
                    return "below_min_similarity"
                return "score_drop_off"
        return "all_included"


class CrossEncoderReRanker:
    """
    High-quality re-ranking using a cross-encoder model.
    
    Cross-encoders jointly encode query and document, providing
    much better relevance scores than bi-encoders at the cost of speed.
    
    Best used after initial retrieval to re-rank top candidates.
    
    NOTE: On CPU, this adds ~500-2000ms latency. Disabled by default.
    Enable only if you have GPU or can tolerate the latency.
    """
    
    def __init__(self, top_k: int = 10):
        """
        Args:
            top_k: Only re-rank top K results for efficiency
        """
        self.top_k = top_k
        self._model = None
        self._executor = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = get_cross_encoder()
        return self._model
    
    def _sync_rerank(self, query: str, to_rerank: List[SearchResult]) -> List[SearchResult]:
        """Synchronous re-ranking (runs in thread pool to avoid blocking)."""
        model = self.model
        if model is None:
            return to_rerank
        
        # Prepare query-document pairs
        pairs = [(query, r.text[:512]) for r in to_rerank]  # Truncate to model max
        
        # Get cross-encoder scores
        scores = model.predict(pairs)
        
        # Update scores
        reranked = []
        for i, result in enumerate(to_rerank):
            reranked.append(SearchResult(
                id=result.id,
                text=result.text,
                score=float(scores[i]),  # Cross-encoder score
                payload={**result.payload, "original_score": result.score},
                source=result.source,
            ))
        
        # Sort by cross-encoder scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked
    
    async def rerank_async(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Re-rank results using cross-encoder (async, runs in thread pool).
        
        This prevents blocking the async event loop during CPU-intensive
        cross-encoder inference.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        if not results:
            return results
        
        model = self.model
        if model is None:
            logger.debug("Cross-encoder not available, skipping re-ranking")
            return results
        
        # Only re-rank top-k for efficiency
        to_rerank = results[:self.top_k]
        remainder = results[self.top_k:]
        
        try:
            # Run CPU-intensive work in thread pool to avoid blocking
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cross-encoder")
            
            loop = asyncio.get_event_loop()
            reranked = await loop.run_in_executor(
                self._executor,
                self._sync_rerank,
                query,
                to_rerank,
            )
            
            # Append remainder (not re-ranked)
            reranked.extend(remainder)
            
            logger.debug(f"Cross-encoder re-ranked {len(to_rerank)} results")
            return reranked
            
        except Exception as e:
            logger.warning(f"Cross-encoder re-ranking failed: {e}, returning original results")
            return results
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Re-rank results using cross-encoder (sync version).
        
        WARNING: This blocks the event loop. Use rerank_async() in async contexts.
        """
        if not results:
            return results
        
        model = self.model
        if model is None:
            logger.debug("Cross-encoder not available, skipping re-ranking")
            return results
        
        # Only re-rank top-k for efficiency
        to_rerank = results[:self.top_k]
        remainder = results[self.top_k:]
        
        try:
            reranked = self._sync_rerank(query, to_rerank)
            reranked.extend(remainder)
            logger.debug(f"Cross-encoder re-ranked {len(to_rerank)} results")
            return reranked
            
        except Exception as e:
            logger.warning(f"Cross-encoder re-ranking failed: {e}, returning original results")
            return results


class ReRanker:
    """
    Re-ranks search results based on additional signals.
    
    Enhanced with:
    - Query term overlap boost
    - Speaker/entity matching boost (increased)
    - Recency boost for recent meetings
    - Exact name match boost
    
    For production, consider using a cross-encoder model for even better results.
    """
    
    def __init__(self):
        self.query_term_boost = 0.15      # Increased from 0.1
        self.recency_boost = 0.08         # Increased from 0.05
        self.speaker_match_boost = 0.15   # Increased from 0.08
        self.exact_match_boost = 0.10     # NEW: boost for exact phrase matches
    
    def _compute_query_overlap(self, query: str, text: str) -> float:
        """Compute query term overlap score."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & text_words)
        return overlap / len(query_words)
    
    def _check_exact_match(self, query: str, text: str) -> bool:
        """Check if query contains exact phrase matches in text."""
        # Extract quoted phrases or multi-word terms
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Check for exact phrase (more than 2 consecutive words from query in text)
        query_words = query_lower.split()
        if len(query_words) >= 2:
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in text_lower:
                    return True
        return False
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entity names (capitalized words)."""
        # Find sequences of capitalized words (potential names)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return [e.lower() for e in entities]
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        boost_speakers: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Re-rank results based on multiple signals.
        
        Args:
            query: Original query
            results: Search results to re-rank
            boost_speakers: Optional speaker names to boost
        
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        query_lower = query.lower()
        query_entities = self._extract_entities(query)
        
        reranked = []
        
        for result in results:
            score = result.score
            
            # Boost for query term overlap
            overlap = self._compute_query_overlap(query, result.text)
            score += overlap * self.query_term_boost
            
            # Boost for exact phrase matches
            if self._check_exact_match(query, result.text):
                score += self.exact_match_boost
            
            # Boost for entity/speaker matches
            if boost_speakers:
                speakers = result.payload.get("speakers", [])
                if any(s.lower() in [bs.lower() for bs in boost_speakers] for s in speakers):
                    score += self.speaker_match_boost
            
            # Boost if query mentions a name that appears in text
            text_entities = self._extract_entities(result.text)
            entity_matches = len(set(query_entities) & set(text_entities))
            score += entity_matches * 0.05
            
            reranked.append(SearchResult(
                id=result.id,
                text=result.text,
                score=score,
                payload=result.payload,
                source=result.source,
            ))
        
        # Re-sort by adjusted scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked
