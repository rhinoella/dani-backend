"""
Tests for hybrid search functionality.
"""

import pytest
from typing import List, Dict, Any

from app.vectorstore.hybrid_search import (
    SearchResult,
    KeywordSearcher,
    HybridSearcher,
    AdaptiveRetriever,
    ReRanker,
)


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        result = SearchResult(
            id="doc_1",
            text="Test document text",
            score=0.85,
            payload={"title": "Test", "date": "2025-01-01"},
            source="vector",
        )
        
        d = result.to_dict()
        
        assert d["id"] == "doc_1"
        assert d["text"] == "Test document text"
        assert d["score"] == 0.85
        assert d["source"] == "vector"
        assert d["title"] == "Test"
        assert d["date"] == "2025-01-01"


class TestKeywordSearcher:
    """Tests for KeywordSearcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.searcher = KeywordSearcher()
        self.documents = [
            {"id": "1", "text": "Alice discussed the Q3 revenue report with Bob"},
            {"id": "2", "text": "The marketing team presented their new campaign"},
            {"id": "3", "text": "Bob mentioned the budget allocation for Q4"},
            {"id": "4", "text": "Alice and Charlie reviewed the product roadmap"},
            {"id": "5", "text": "Technical debt discussion led by the engineering team"},
        ]
    
    def test_basic_search(self):
        """Should find documents matching keywords."""
        results = self.searcher.search("Alice revenue report", self.documents)
        
        assert len(results) > 0
        # Document mentioning Alice and revenue should rank high
        assert results[0].id == "1"
    
    def test_search_no_matches(self):
        """Should return empty for no matches."""
        results = self.searcher.search("xyz123 nonexistent", self.documents)
        assert len(results) == 0
    
    def test_search_stop_words_ignored(self):
        """Should ignore stop words in query."""
        results1 = self.searcher.search("the revenue report", self.documents)
        results2 = self.searcher.search("revenue report", self.documents)
        
        # Results should be similar regardless of stop words
        if results1 and results2:
            assert results1[0].id == results2[0].id
    
    def test_search_limit(self):
        """Should respect limit parameter."""
        results = self.searcher.search("team", self.documents, limit=2)
        assert len(results) <= 2
    
    def test_search_case_insensitive(self):
        """Should be case insensitive."""
        results_lower = self.searcher.search("alice", self.documents)
        results_upper = self.searcher.search("ALICE", self.documents)
        
        assert len(results_lower) == len(results_upper)
        if results_lower:
            assert results_lower[0].id == results_upper[0].id


class TestHybridSearcher:
    """Tests for HybridSearcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.searcher = HybridSearcher(
            vector_weight=0.7,
            keyword_weight=0.3,
        )
    
    def test_reciprocal_rank_fusion_single_list(self):
        """Should handle single result list."""
        results = [
            SearchResult(id="1", text="doc1", score=0.9, payload={}),
            SearchResult(id="2", text="doc2", score=0.8, payload={}),
        ]
        
        merged = self.searcher.reciprocal_rank_fusion([results])
        
        assert len(merged) == 2
        # Order should be preserved
        assert merged[0].id == "1"
        assert merged[1].id == "2"
    
    def test_reciprocal_rank_fusion_merge(self):
        """Should merge multiple lists using RRF."""
        list1 = [
            SearchResult(id="1", text="doc1", score=0.9, payload={}),
            SearchResult(id="2", text="doc2", score=0.8, payload={}),
        ]
        list2 = [
            SearchResult(id="2", text="doc2", score=0.85, payload={}),
            SearchResult(id="3", text="doc3", score=0.75, payload={}),
        ]
        
        merged = self.searcher.reciprocal_rank_fusion([list1, list2])
        
        # Doc 2 appears in both lists, should rank higher
        assert any(r.id == "2" for r in merged)
        # All unique docs should be present
        ids = {r.id for r in merged}
        assert ids == {"1", "2", "3"}
    
    def test_reciprocal_rank_fusion_weighted(self):
        """Should respect weights in RRF."""
        list1 = [SearchResult(id="1", text="doc1", score=0.9, payload={})]
        list2 = [SearchResult(id="2", text="doc2", score=0.9, payload={})]
        
        # Heavily weight list1
        merged = self.searcher.reciprocal_rank_fusion(
            [list1, list2],
            weights=[0.9, 0.1],
        )
        
        # Doc from list1 should rank higher due to higher weight
        assert merged[0].id == "1"
    
    def test_search_hybrid(self):
        """Should combine vector and keyword results."""
        vector_results = [
            SearchResult(id="1", text="Semantic match", score=0.9, payload={}),
        ]
        documents = [
            {"id": "1", "text": "Semantic match"},
            {"id": "2", "text": "Keyword match revenue report"},
        ]
        
        results = self.searcher.search(
            query="revenue report",
            vector_results=vector_results,
            all_documents=documents,
            limit=5,
        )
        
        assert len(results) > 0
        assert all(r.source == "hybrid" for r in results)


class TestAdaptiveRetriever:
    """Tests for AdaptiveRetriever class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.retriever = AdaptiveRetriever(
            min_similarity=0.65,
            max_chunks=15,
            min_chunks=3,
            drop_off_threshold=0.15,
        )
    
    def test_min_chunks_preserved(self):
        """Should preserve minimum number of chunks."""
        results = [
            SearchResult(id=str(i), text=f"doc{i}", score=0.9 - i*0.1, payload={})
            for i in range(10)
        ]
        
        filtered, meta = self.retriever.filter_results(results)
        
        assert len(filtered) >= 3  # min_chunks
    
    def test_max_chunks_respected(self):
        """Should not exceed maximum chunks."""
        results = [
            SearchResult(id=str(i), text=f"doc{i}", score=0.9, payload={})
            for i in range(20)
        ]
        
        filtered, meta = self.retriever.filter_results(results)
        
        assert len(filtered) <= 15  # max_chunks
    
    def test_min_similarity_cutoff(self):
        """Should cut off results below minimum similarity."""
        results = [
            SearchResult(id="1", text="doc1", score=0.9, payload={}),
            SearchResult(id="2", text="doc2", score=0.8, payload={}),
            SearchResult(id="3", text="doc3", score=0.7, payload={}),
            SearchResult(id="4", text="doc4", score=0.5, payload={}),  # Below threshold
            SearchResult(id="5", text="doc5", score=0.4, payload={}),
        ]
        
        filtered, meta = self.retriever.filter_results(results)
        
        # Should not include scores below 0.65
        assert all(r.score >= 0.65 for r in filtered)
    
    def test_drop_off_detection(self):
        """Should detect large score drop-offs."""
        results = [
            SearchResult(id="1", text="doc1", score=0.9, payload={}),
            SearchResult(id="2", text="doc2", score=0.88, payload={}),
            SearchResult(id="3", text="doc3", score=0.86, payload={}),
            SearchResult(id="4", text="doc4", score=0.65, payload={}),  # Big drop
            SearchResult(id="5", text="doc5", score=0.64, payload={}),
        ]
        
        filtered, meta = self.retriever.filter_results(results)
        
        # Should stop at the big drop-off (after min_chunks)
        assert len(filtered) <= 4
    
    def test_empty_results(self):
        """Should handle empty results."""
        filtered, meta = self.retriever.filter_results([])
        
        assert filtered == []
        assert meta["reason"] == "no_results"
    
    def test_metadata_returned(self):
        """Should return useful metadata."""
        results = [
            SearchResult(id="1", text="doc1", score=0.9, payload={}),
            SearchResult(id="2", text="doc2", score=0.8, payload={}),
        ]
        
        filtered, meta = self.retriever.filter_results(results)
        
        assert "total_candidates" in meta
        assert "filtered_count" in meta
        assert "top_score" in meta
        assert "cutoff_reason" in meta


class TestReRanker:
    """Tests for ReRanker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reranker = ReRanker()
    
    def test_query_overlap_boost(self):
        """Should boost results with query term overlap."""
        results = [
            SearchResult(
                id="1", 
                text="General meeting notes about various topics",
                score=0.9, 
                payload={}
            ),
            SearchResult(
                id="2", 
                text="Revenue report discussion with detailed numbers",
                score=0.85, 
                payload={}
            ),
        ]
        
        reranked = self.reranker.rerank("revenue report", results)
        
        # Doc 2 should be boosted due to keyword overlap
        assert reranked[0].id == "2"
    
    def test_speaker_boost(self):
        """Should boost results with matching speakers."""
        results = [
            SearchResult(
                id="1", 
                text="General discussion",
                score=0.9, 
                payload={"speakers": ["Charlie", "Dave"]}
            ),
            SearchResult(
                id="2", 
                text="Another discussion",
                score=0.85, 
                payload={"speakers": ["Alice", "Bob"]}
            ),
        ]
        
        reranked = self.reranker.rerank(
            "What did Alice say?", 
            results,
            boost_speakers=["Alice"],
        )
        
        # Doc 2 should be boosted due to speaker match
        assert reranked[0].id == "2"
    
    def test_empty_results(self):
        """Should handle empty results."""
        reranked = self.reranker.rerank("test query", [])
        assert reranked == []
    
    def test_preserves_all_results(self):
        """Should preserve all results after reranking."""
        results = [
            SearchResult(id=str(i), text=f"doc{i}", score=0.9, payload={})
            for i in range(5)
        ]
        
        reranked = self.reranker.rerank("test", results)
        
        assert len(reranked) == len(results)
        assert {r.id for r in reranked} == {r.id for r in results}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
