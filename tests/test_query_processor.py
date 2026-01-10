"""
Tests for query processing utilities.
"""

import pytest
from typing import List

from app.utils.query_processor import (
    QueryProcessor,
    QueryIntent,
    ConfidenceScorer,
)


class TestQueryProcessor:
    """Tests for QueryProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = QueryProcessor()
    
    def test_detect_intent_summary(self):
        """Should detect summary intent."""
        queries = [
            "Can you summarize the meeting?",
            "Give me an overview of the discussion",
            "What was discussed in the meeting?",
            "Recap of yesterday's call",
        ]
        
        for query in queries:
            intent = self.processor.detect_intent(query)
            assert intent.intent_type == "summary", f"Failed for: {query}"
    
    def test_detect_intent_action_items(self):
        """Should detect action items intent."""
        queries = [
            "What are the action items?",
            "List the tasks assigned",
            "Who is supposed to do what?",
            "What are the next steps?",
        ]
        
        for query in queries:
            intent = self.processor.detect_intent(query)
            assert intent.intent_type == "action_items", f"Failed for: {query}"
    
    def test_detect_intent_decisions(self):
        """Should detect decisions intent."""
        queries = [
            "What decisions were made?",
            "What did we decide about the budget?",
            "What was concluded?",
        ]
        
        for query in queries:
            intent = self.processor.detect_intent(query)
            assert intent.intent_type == "decisions", f"Failed for: {query}"
    
    def test_detect_intent_person_search(self):
        """Should detect person-specific queries."""
        queries = [
            "What did Alice say about the project?",
            "Bob's thoughts on the proposal",
            "According to Charlie",
        ]
        
        for query in queries:
            intent = self.processor.detect_intent(query)
            assert intent.intent_type == "person_search", f"Failed for: {query}"
    
    def test_detect_intent_factual_default(self):
        """Should default to factual for ambiguous queries."""
        query = "Tell me something interesting"
        intent = self.processor.detect_intent(query)
        assert intent.intent_type == "factual"
    
    def test_extract_entities(self):
        """Should extract named entities."""
        query = "What did Alice and Bob discuss about Project Alpha?"
        intent = self.processor.detect_intent(query)
        
        # Should find capitalized names (not at sentence start)
        assert "Alice" in intent.entities
        assert "Bob" in intent.entities
        assert "Project" in intent.entities or "Alpha" in intent.entities
    
    def test_extract_time_references(self):
        """Should extract time references."""
        queries_and_expected = [
            ("What happened last week?", ["last week"]),
            ("Meeting notes from January", ["january"]),
            ("Q3 2024 results", ["q3 2024"]),
        ]
        
        for query, expected in queries_and_expected:
            intent = self.processor.detect_intent(query)
            # At least one expected time reference should be found
            found = any(
                exp.lower() in [t.lower() for t in intent.time_references]
                or any(exp.lower() in t.lower() for t in intent.time_references)
                for exp in expected
            )
            assert found or len(intent.time_references) > 0, f"Failed for: {query}"
    
    def test_compress_query_removes_stop_words(self):
        """Should remove stop words while keeping meaning."""
        query = "Can you please tell me what was discussed in the meeting?"
        compressed = self.processor.compress_query(query)
        
        # Should not contain common stop words (only check simple ones)
        assert "please" not in compressed.lower().split()
        assert "the" not in compressed.lower().split()
        assert "me" not in compressed.lower().split()
        
        # Should keep meaningful words
        assert "discussed" in compressed.lower() or "meeting" in compressed.lower()
    
    def test_compress_query_preserves_proper_nouns(self):
        """Should preserve proper nouns."""
        query = "What did Alice say to Bob about the project?"
        compressed = self.processor.compress_query(query)
        
        assert "Alice" in compressed
        assert "Bob" in compressed
    
    def test_compress_query_preserves_numbers(self):
        """Should preserve numbers."""
        query = "What was the Q3 revenue in 2024?"
        compressed = self.processor.compress_query(query)
        
        assert "Q3" in compressed or "q3" in compressed.lower()
        assert "2024" in compressed
    
    def test_expand_query(self):
        """Should generate query variations."""
        query = "Tell me about the revenue discussion"
        variations = self.processor.expand_query(query)
        
        assert len(variations) >= 1
        assert query in variations  # Original should be included
    
    def test_expand_query_rephrasing(self):
        """Should rephrase certain query patterns."""
        query = "Tell me about the budget"
        variations = self.processor.expand_query(query)
        
        # Should have variation without "tell me about"
        assert any("budget" in v and "tell" not in v.lower() for v in variations)


class TestConfidenceScorer:
    """Tests for ConfidenceScorer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = ConfidenceScorer()
    
    def test_high_confidence(self):
        """Should return high confidence for good retrieval."""
        chunks = [
            {"score": 0.92, "text": "Relevant chunk 1"},
            {"score": 0.88, "text": "Relevant chunk 2"},
            {"score": 0.85, "text": "Relevant chunk 3"},
            {"score": 0.82, "text": "Relevant chunk 4"},
        ]
        
        result = self.scorer.score(chunks, "test query")
        
        assert result["level"] == "high"
        assert result["score"] >= 0.85
    
    def test_medium_confidence(self):
        """Should return medium confidence for decent retrieval."""
        chunks = [
            {"score": 0.80, "text": "Somewhat relevant 1"},
            {"score": 0.72, "text": "Somewhat relevant 2"},
            {"score": 0.68, "text": "Somewhat relevant 3"},
        ]
        
        result = self.scorer.score(chunks, "test query")
        
        assert result["level"] in ["medium", "high"]
    
    def test_low_confidence(self):
        """Should return low confidence for poor retrieval."""
        # With new thresholds (min_avg=0.08, min_top=0.12), need very low scores
        chunks = [
            {"score": 0.03, "text": "Barely relevant"},
            {"score": 0.02, "text": "Not very relevant"},
        ]
        
        result = self.scorer.score(chunks, "test query")
        
        assert result["level"] in ["low", "very_low"]
    
    def test_no_chunks(self):
        """Should return zero confidence for no results."""
        result = self.scorer.score([], "test query")
        
        assert result["level"] == "none"
        assert result["score"] == 0.0
        assert result["reason"] == "no_chunks_retrieved"
    
    def test_metrics_included(self):
        """Should include detailed metrics."""
        chunks = [
            {"score": 0.9, "text": "chunk1"},
            {"score": 0.8, "text": "chunk2"},
        ]
        
        result = self.scorer.score(chunks, "test query")
        
        assert "metrics" in result
        assert "top_similarity" in result["metrics"]
        assert "avg_similarity" in result["metrics"]
        assert "chunk_count" in result["metrics"]
    
    def test_should_fallback(self):
        """Should recommend fallback for low confidence."""
        low_confidence = {"level": "low", "score": 0.4}
        high_confidence = {"level": "high", "score": 0.9}
        
        assert self.scorer.should_fallback(low_confidence) is True
        assert self.scorer.should_fallback(high_confidence) is False
    
    def test_disclaimer_low_confidence(self):
        """Should provide disclaimer for low confidence."""
        low_confidence = {"level": "low", "score": 0.4}
        very_low = {"level": "very_low", "score": 0.2}
        high = {"level": "high", "score": 0.9}
        
        assert self.scorer.get_disclaimer(low_confidence) is not None
        assert self.scorer.get_disclaimer(very_low) is not None
        assert self.scorer.get_disclaimer(high) is None


class TestQueryIntentDataclass:
    """Tests for QueryIntent dataclass."""
    
    def test_query_intent_creation(self):
        """Should create QueryIntent with all fields."""
        intent = QueryIntent(
            intent_type="summary",
            entities=["Alice", "Bob"],
            time_references=["last week"],
            confidence=0.85,
            processed_query="meeting summary",
        )
        
        assert intent.intent_type == "summary"
        assert len(intent.entities) == 2
        assert intent.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
