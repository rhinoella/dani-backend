"""Tests for Document Type Filter in retrieval."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.retrieval import MetadataFilter, DocSourceType
from app.services.retrieval_service import RetrievalService


class TestDocTypeFilter:
    """Tests for document type filtering."""

    def test_metadata_filter_accepts_doc_type(self):
        """MetadataFilter should accept doc_type field."""
        filter = MetadataFilter(doc_type="meeting")
        assert filter.doc_type == "meeting"

    def test_metadata_filter_all_doc_types(self):
        """Should accept all valid doc_type values."""
        valid_types = ["meeting", "email", "document", "note", "all"]
        
        for doc_type in valid_types:
            filter = MetadataFilter(doc_type=doc_type)
            assert filter.doc_type == doc_type

    def test_metadata_filter_doc_type_optional(self):
        """doc_type should be optional."""
        filter = MetadataFilter()
        assert filter.doc_type is None

    def test_metadata_filter_combined_with_other_fields(self):
        """Should work with other filter fields."""
        filter = MetadataFilter(
            doc_type="meeting",
            speakers=["Bunmi", "Sarah"],
            date_from=1704067200000,
        )
        
        assert filter.doc_type == "meeting"
        assert filter.speakers == ["Bunmi", "Sarah"]
        assert filter.date_from == 1704067200000


class TestRetrievalServiceDocTypeFilter:
    """Tests for doc_type filter in RetrievalService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = RetrievalService()

    def test_build_filter_with_doc_type(self):
        """Should build Qdrant filter with doc_type condition."""
        from qdrant_client.http import models as qm
        
        metadata = MetadataFilter(doc_type="meeting")
        filter_ = self.service._build_filter(metadata)
        
        assert filter_ is not None
        assert len(filter_.must) == 1
        
        condition = filter_.must[0]
        assert condition.key == "doc_type"
        assert condition.match.value == "meeting"

    def test_build_filter_doc_type_all_skipped(self):
        """doc_type='all' should not create a filter condition."""
        metadata = MetadataFilter(doc_type="all")
        filter_ = self.service._build_filter(metadata)
        
        # No filter should be created for 'all'
        assert filter_ is None

    def test_build_filter_combined_conditions(self):
        """Should combine doc_type with other conditions."""
        metadata = MetadataFilter(
            doc_type="email",
            organizer_email="bunmi@example.com",
        )
        filter_ = self.service._build_filter(metadata)
        
        assert filter_ is not None
        assert len(filter_.must) == 2
        
        # Check both conditions exist
        keys = [cond.key for cond in filter_.must]
        assert "doc_type" in keys
        assert "organizer_email" in keys


class TestChatServiceDocTypeFilter:
    """Tests for doc_type filter in ChatService."""

    @pytest.mark.asyncio
    async def test_answer_passes_doc_type_to_retrieval(self):
        """ChatService.answer() should pass doc_type to retrieval."""
        from app.services.chat_service import ChatService
        
        service = ChatService()
        
        # Mock retrieval
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "low", "metrics": {}},
            "query_analysis": {"intent": "factual", "entities": [], "time_references": [], "processed_query": "test"},
            "disclaimer": None,
        })
        
        await service.answer(
            query="What happened in the last meeting?",
            doc_type="meeting",
        )
        
        # Verify retrieval was called with metadata_filter
        call_args = service.retrieval.search_with_confidence.call_args
        metadata_filter = call_args[1].get("metadata_filter")
        
        assert metadata_filter is not None
        assert metadata_filter.doc_type == "meeting"

    @pytest.mark.asyncio
    async def test_answer_no_filter_when_doc_type_all(self):
        """Should not create filter when doc_type is 'all'."""
        from app.services.chat_service import ChatService
        
        service = ChatService()
        
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "low", "metrics": {}},
            "query_analysis": {"intent": "factual", "entities": [], "time_references": [], "processed_query": "test"},
            "disclaimer": None,
        })
        
        await service.answer(
            query="What happened?",
            doc_type="all",
        )
        
        call_args = service.retrieval.search_with_confidence.call_args
        metadata_filter = call_args[1].get("metadata_filter")
        
        # 'all' should result in no filter
        assert metadata_filter is None

    @pytest.mark.asyncio
    async def test_answer_no_filter_when_doc_type_none(self):
        """Should not create filter when doc_type is None."""
        from app.services.chat_service import ChatService
        
        service = ChatService()
        
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "low", "metrics": {}},
            "query_analysis": {"intent": "factual", "entities": [], "time_references": [], "processed_query": "test"},
            "disclaimer": None,
        })
        
        await service.answer(
            query="What happened?",
            doc_type=None,
        )
        
        call_args = service.retrieval.search_with_confidence.call_args
        metadata_filter = call_args[1].get("metadata_filter")
        
        assert metadata_filter is None


class TestChatRequestDocType:
    """Tests for doc_type in chat request schema."""

    def test_chat_request_has_doc_type(self):
        """ChatRequest schema should have doc_type field."""
        from app.schemas.chat import ChatRequest
        
        assert "doc_type" in ChatRequest.model_fields

    def test_chat_request_accepts_doc_type(self):
        """Should accept doc_type in request."""
        from app.schemas.chat import ChatRequest
        
        request = ChatRequest(
            query="What was discussed in meetings?",
            doc_type="meeting",
        )
        
        assert request.doc_type == "meeting"
