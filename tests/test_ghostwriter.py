"""Tests for GhostwriterService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ghostwriter_service import (
    GhostwriterService, 
    ContentType, 
    CONTENT_TEMPLATES,
    get_ghostwriter,
)


class TestGhostwriterService:
    """Tests for GhostwriterService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = GhostwriterService()

    def test_content_types_enum(self):
        """Should have all expected content types."""
        expected_types = [
            "linkedin_post",
            "email", 
            "blog_draft",
            "tweet_thread",
            "newsletter",
            "meeting_summary",
        ]
        
        actual_types = [ct.value for ct in ContentType]
        assert set(expected_types) == set(actual_types)

    def test_all_content_types_have_templates(self):
        """Every content type should have a corresponding template."""
        for ct in ContentType:
            assert ct in CONTENT_TEMPLATES, f"Missing template for {ct.value}"
            assert len(CONTENT_TEMPLATES[ct]) > 100, f"Template for {ct.value} seems too short"

    def test_get_content_types(self):
        """Should return list of content types with descriptions."""
        types = self.service.get_content_types()
        
        assert len(types) == len(ContentType)
        for t in types:
            assert "type" in t
            assert "description" in t
            assert len(t["description"]) > 10

    @pytest.mark.asyncio
    async def test_generate_invalid_content_type(self):
        """Should return error for invalid content type."""
        # Use a string that's not a valid ContentType
        result = await self.service.generate(
            content_type="invalid_type",  # type: ignore
            request="Write something",
        )
        
        assert "error" in result
        assert "supported_types" in result

    @pytest.mark.asyncio
    async def test_generate_linkedin_post(self):
        """Should generate LinkedIn post with mocked LLM."""
        # Mock the dependencies
        self.service.llm.generate = AsyncMock(return_value="Just launched our mobile app! 🚀")
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [
                {"title": "Mobile Launch Meeting", "text": "Launching mobile app Q1", "score": 0.9}
            ],
            "confidence": {"level": "high", "metrics": {}},
        })

        result = await self.service.generate(
            content_type=ContentType.LINKEDIN_POST,
            request="Write about our mobile app launch",
        )

        assert "error" not in result
        assert result["content"] == "Just launched our mobile app! 🚀"
        assert result["content_type"] == "linkedin_post"
        assert "word_count" in result
        assert "sources" in result
        assert "timing" in result

    @pytest.mark.asyncio
    async def test_generate_email(self):
        """Should generate email with mocked LLM."""
        mock_email = """Subject: Q1 Strategy Update

Hi Team,

Our mobile app is on track for Q1 launch. Key milestones have been met.

Best,
DANI"""
        
        self.service.llm.generate = AsyncMock(return_value=mock_email)
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [{"title": "Strategy Meeting", "text": "Mobile on track", "score": 0.85}],
            "confidence": {"level": "medium", "metrics": {}},
        })

        result = await self.service.generate(
            content_type=ContentType.EMAIL,
            request="Update the team on Q1 progress",
        )

        assert "error" not in result
        assert "Subject:" in result["content"]
        assert result["content_type"] == "email"

    @pytest.mark.asyncio
    async def test_generate_with_doc_type_filter(self):
        """Should pass doc_type filter to retrieval (non-meeting types)."""
        self.service.llm.generate = AsyncMock(return_value="Generated content")
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "low", "metrics": {}},
        })

        # Use 'document' type since 'meeting' has a workaround that sets it to 'all'
        await self.service.generate(
            content_type=ContentType.LINKEDIN_POST,
            request="Write about documents",
            doc_type="document",
        )

        # Verify search was called with metadata filter
        call_args = self.service.retrieval.search_with_confidence.call_args
        assert call_args[1].get("metadata_filter") is not None

    @pytest.mark.asyncio
    async def test_generate_with_tone(self):
        """Should include tone in request."""
        self.service.llm.generate = AsyncMock(return_value="Formal content")
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "low", "metrics": {}},
        })

        result = await self.service.generate(
            content_type=ContentType.EMAIL,
            request="Draft an email",
            tone="formal",
        )

        assert result["metadata"]["tone"] == "formal"

    @pytest.mark.asyncio
    async def test_generate_handles_llm_error(self):
        """Should return error if LLM fails."""
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "low", "metrics": {}},
        })
        self.service.llm.generate = AsyncMock(side_effect=Exception("LLM failed"))

        result = await self.service.generate(
            content_type=ContentType.LINKEDIN_POST,
            request="Write something",
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_refine_content(self):
        """Should refine content based on feedback."""
        self.service.llm.generate = AsyncMock(return_value="Refined and shorter post!")

        result = await self.service.refine(
            content="Original long post about mobile launch that needs improvement",
            feedback="Make it shorter and punchier",
            content_type=ContentType.LINKEDIN_POST,
        )

        assert "error" not in result
        assert result["content"] == "Refined and shorter post!"
        assert result["feedback_applied"] == "Make it shorter and punchier"

    @pytest.mark.asyncio
    async def test_refine_handles_llm_error(self):
        """Should return error if refinement fails."""
        self.service.llm.generate = AsyncMock(side_effect=Exception("Refinement failed"))

        result = await self.service.refine(
            content="Some content",
            feedback="Make changes",
            content_type=ContentType.EMAIL,
        )

        assert "error" in result


class TestGhostwriterTemplates:
    """Tests for content templates."""

    def test_linkedin_template_has_key_elements(self):
        """LinkedIn template should guide proper post structure."""
        template = CONTENT_TEMPLATES[ContentType.LINKEDIN_POST]
        
        assert "hook" in template.lower()
        assert "hashtag" in template.lower()
        assert "{context}" in template
        assert "{request}" in template

    def test_email_template_has_key_elements(self):
        """Email template should include subject line guidance."""
        template = CONTENT_TEMPLATES[ContentType.EMAIL]
        
        assert "subject" in template.lower()
        assert "greeting" in template.lower()
        assert "{context}" in template
        assert "{request}" in template

    def test_blog_template_has_structure(self):
        """Blog template should guide longer-form content."""
        template = CONTENT_TEMPLATES[ContentType.BLOG_DRAFT]
        
        assert "header" in template.lower()
        assert "500" in template or "800" in template  # Word count guidance
        assert "{context}" in template

    def test_tweet_thread_template_has_constraints(self):
        """Tweet thread template should mention character limits."""
        template = CONTENT_TEMPLATES[ContentType.TWEET_THREAD]
        
        assert "280" in template  # Character limit
        assert "thread" in template.lower()


class TestGhostwriterSingleton:
    """Tests for singleton pattern."""

    def test_get_ghostwriter_returns_same_instance(self):
        """Should return the same instance on repeated calls."""
        instance1 = get_ghostwriter()
        instance2 = get_ghostwriter()
        
        assert instance1 is instance2

    def test_get_ghostwriter_is_initialized(self):
        """Should return a properly initialized service."""
        service = get_ghostwriter()
        
        assert service.llm is not None
        assert service.retrieval is not None
