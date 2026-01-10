"""
Tests for Pydantic schemas.
"""

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, ChatResponse


class TestChatRequest:
    """Tests for ChatRequest schema."""
    
    def test_minimal_request(self):
        """Test creating request with just query."""
        request = ChatRequest(query="What is the capital of France?")
        assert request.query == "What is the capital of France?"
        assert request.verbose is False
        assert request.output_format is None
        assert request.conversation_id is None
        assert request.include_history is True
    
    def test_full_request(self):
        """Test creating request with all fields."""
        request = ChatRequest(
            query="Summarize the meeting",
            verbose=True,
            output_format="summary",
            conversation_id="conv-123",
            include_history=False
        )
        assert request.query == "Summarize the meeting"
        assert request.verbose is True
        assert request.output_format == "summary"
        assert request.conversation_id == "conv-123"
        assert request.include_history is False
    
    def test_output_format_values(self):
        """Test valid output format values."""
        valid_formats = [
            "summary", "decisions", "tasks", "insights",
            "email", "whatsapp", "slides", "infographic"
        ]
        for fmt in valid_formats:
            request = ChatRequest(query="test", output_format=fmt)
            assert request.output_format == fmt
    
    def test_invalid_output_format(self):
        """Test invalid output format raises error."""
        with pytest.raises(ValidationError):
            ChatRequest(query="test", output_format="invalid_format")
    
    def test_empty_query(self):
        """Test that empty query is allowed (validation at route level)."""
        request = ChatRequest(query="")
        assert request.query == ""
    
    def test_missing_query_raises_error(self):
        """Test that missing query raises error."""
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:
    """Tests for ChatResponse schema."""
    
    def test_minimal_response(self):
        """Test creating response with required fields only."""
        response = ChatResponse(answer="Paris", sources=[])
        assert response.answer == "Paris"
        assert response.sources == []
        assert response.output_format is None
        assert response.debug is None
        assert response.conversation_id is None
        assert response.message_id is None
        assert response.confidence is None
    
    def test_full_response(self):
        """Test creating response with all fields."""
        sources = [{"document": "doc1.pdf", "page": 5}]
        debug_info = {"latency_ms": 150}
        
        response = ChatResponse(
            answer="The answer is 42",
            sources=sources,
            output_format="summary",
            debug=debug_info,
            conversation_id="conv-456",
            message_id="msg-789",
            confidence=0.95
        )
        
        assert response.answer == "The answer is 42"
        assert response.sources == sources
        assert response.output_format == "summary"
        assert response.debug == debug_info
        assert response.conversation_id == "conv-456"
        assert response.message_id == "msg-789"
        assert response.confidence == 0.95
    
    def test_response_with_empty_sources(self):
        """Test response with empty sources list."""
        response = ChatResponse(answer="No sources found", sources=[])
        assert response.sources == []
    
    def test_response_with_multiple_sources(self):
        """Test response with multiple sources."""
        sources = [
            {"document": "doc1.pdf", "page": 1, "score": 0.9},
            {"document": "doc2.pdf", "page": 10, "score": 0.8},
            {"document": "meeting.txt", "line": 50, "score": 0.75},
        ]
        response = ChatResponse(answer="Found in multiple docs", sources=sources)
        assert len(response.sources) == 3
    
    def test_confidence_bounds(self):
        """Test confidence values at boundaries."""
        # 0 confidence
        response = ChatResponse(answer="test", sources=[], confidence=0.0)
        assert response.confidence == 0.0
        
        # 1.0 confidence
        response = ChatResponse(answer="test", sources=[], confidence=1.0)
        assert response.confidence == 1.0
        
        # None confidence (default)
        response = ChatResponse(answer="test", sources=[])
        assert response.confidence is None
