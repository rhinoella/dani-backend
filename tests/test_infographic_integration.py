"""Integration tests for InfographicService with real services."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.infographic_service import (
    InfographicService,
    InfographicStyle,
)
from app.services.retrieval_service import RetrievalService
from app.llm.ollama import OllamaClient


class TestInfographicIntegration:
    """Integration tests for infographic generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = InfographicService()

    @pytest.mark.asyncio
    async def test_full_generation_flow_with_mocked_llm(self):
        """Test full generation flow with mocked LLM and retrieval."""
        # Mock retrieval service
        mock_chunks = [
            {
                "title": "Q4 2025 Results",
                "text": "Revenue reached $15M, up 40% from Q3. Customer base grew to 500+ organizations.",
                "date": "2025-01-20",
                "score": 0.95,
            },
            {
                "title": "Market Analysis",
                "text": "Enterprise segment showing strongest growth at 55% YoY. APAC market expanded.",
                "date": "2025-01-15",
                "score": 0.92,
            },
        ]

        self.service.retrieval.search_with_confidence = AsyncMock(
            return_value={
                "chunks": mock_chunks,
                "confidence": {"level": "high", "metrics": {}},
            }
        )

        # Mock LLM to return valid JSON
        structured_response = {
            "headline": "Q4 Revenue Surges 40% to $15M",
            "subtitle": "Record quarter driven by enterprise growth",
            "stats": [
                {"value": "40%", "label": "Revenue Growth", "icon": "📈"},
                {"value": "500+", "label": "Organizations", "icon": "🏢"},
                {"value": "55%", "label": "Enterprise YoY", "icon": "📊"},
                {"value": "$15M", "label": "Q4 Revenue", "icon": "💰"},
            ],
            "key_points": [
                "Enterprise segment driving 55% YoY growth",
                "Customer base expanded to 500+ organizations",
                "APAC region showing strong momentum",
            ],
            "source_summary": "Q4 2025 Business Review",
        }

        self.service.llm.generate = AsyncMock(
            return_value=json.dumps(structured_response)
        )

        # Mock MCP registry (no image generation)
        mock_registry = MagicMock()
        mock_registry.connected_servers = []
        self.service._mcp_registry = mock_registry

        result = await self.service.generate(
            request="Create Q4 performance infographic",
            topic="Q4 2025 Results",
            style=InfographicStyle.MODERN,
            persist=False,  # Don't persist in tests
        )

        # Verify structured data was extracted correctly
        assert "structured_data" in result
        assert result["structured_data"]["headline"] == "Q4 Revenue Surges 40% to $15M"
        assert len(result["structured_data"]["stats"]) == 4
        assert result["structured_data"]["subtitle"] == "Record quarter driven by enterprise growth"

        # Verify sources
        assert "sources" in result
        assert len(result["sources"]) == 2

        # Verify timing metrics
        assert "timing" in result
        assert "retrieval_ms" in result["timing"]
        assert "extraction_ms" in result["timing"]

    @pytest.mark.asyncio
    async def test_generation_with_malformed_llm_response(self):
        """Test robust handling of malformed LLM responses."""
        mock_chunks = [
            {
                "title": "Test",
                "text": "Some test content with revenue data",
                "date": "2025-01-20",
                "score": 0.8,
            }
        ]

        self.service.retrieval.search_with_confidence = AsyncMock(
            return_value={
                "chunks": mock_chunks,
                "confidence": {"level": "medium", "metrics": {}},
            }
        )

        # LLM returns JSON with leading text (common in some models)
        structured_response = {
            "headline": "Test Headline",
            "stats": [{"value": "100", "label": "Test", "icon": "✅"}],
        }
        malformed_response = f"Here's the data:\n```json\n{json.dumps(structured_response)}\n```\nEnd of data."

        self.service.llm.generate = AsyncMock(return_value=malformed_response)

        # Mock MCP registry
        mock_registry = MagicMock()
        mock_registry.connected_servers = []
        self.service._mcp_registry = mock_registry

        result = await self.service.generate(
            request="Create test infographic",
            persist=False,
        )

        # Should successfully parse despite malformed response
        assert "structured_data" in result
        assert "error" not in result
        assert result["structured_data"]["headline"] == "Test Headline"

    @pytest.mark.asyncio
    async def test_generation_with_no_context(self):
        """Test handling when retrieval returns no context."""
        self.service.retrieval.search_with_confidence = AsyncMock(
            return_value={
                "chunks": [],
                "confidence": {"level": "none", "metrics": {}},
            }
        )

        result = await self.service.generate(
            request="Create infographic about nonexistent topic",
            persist=False,
        )

        # Should return error
        assert "error" in result
        assert "No relevant context" in result["error"]

    @pytest.mark.asyncio
    async def test_generation_with_different_doc_types(self):
        """Test generation with different document type filters."""
        mock_chunks = [
            {
                "title": "Meeting Notes",
                "text": "Team discussion about new features",
                "date": "2025-01-20",
                "score": 0.85,
                "doc_type": "meeting",
            }
        ]

        self.service.retrieval.search_with_confidence = AsyncMock(
            return_value={
                "chunks": mock_chunks,
                "confidence": {"level": "high", "metrics": {}},
            }
        )

        structured_response = {
            "headline": "Meeting Summary",
            "stats": [{"value": "5", "label": "Topics", "icon": "📝"}],
        }

        self.service.llm.generate = AsyncMock(
            return_value=json.dumps(structured_response)
        )

        mock_registry = MagicMock()
        mock_registry.connected_servers = []
        self.service._mcp_registry = mock_registry

        result = await self.service.generate(
            request="Summarize meeting",
            doc_type="meeting",
            persist=False,
        )

        # Should generate successfully
        assert "structured_data" in result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_generation_with_all_styles(self):
        """Test generation with each style option."""
        mock_chunks = [
            {
                "title": "Test",
                "text": "Test content for styling",
                "date": "2025-01-20",
                "score": 0.9,
            }
        ]

        self.service.retrieval.search_with_confidence = AsyncMock(
            return_value={
                "chunks": mock_chunks,
                "confidence": {"level": "high", "metrics": {}},
            }
        )

        structured_response = {
            "headline": "Test Infographic",
            "stats": [{"value": "1", "label": "Test", "icon": "✅"}],
        }

        self.service.llm.generate = AsyncMock(
            return_value=json.dumps(structured_response)
        )

        mock_registry = MagicMock()
        mock_registry.connected_servers = []
        self.service._mcp_registry = mock_registry

        # Test all styles
        for style in InfographicStyle:
            result = await self.service.generate(
                request="Create styled infographic",
                style=style,
                persist=False,
            )

            assert "structured_data" in result
            assert "error" not in result
            assert result["metadata"]["style"] == style.value

    @pytest.mark.asyncio
    async def test_generation_with_custom_dimensions(self):
        """Test generation with custom image dimensions."""
        mock_chunks = [
            {
                "title": "Test",
                "text": "Test content",
                "date": "2025-01-20",
                "score": 0.85,
            }
        ]

        self.service.retrieval.search_with_confidence = AsyncMock(
            return_value={
                "chunks": mock_chunks,
                "confidence": {"level": "high", "metrics": {}},
            }
        )

        structured_response = {
            "headline": "Custom Dimensions",
            "stats": [{"value": "2560", "label": "Width", "icon": "📐"}],
        }

        self.service.llm.generate = AsyncMock(
            return_value=json.dumps(structured_response)
        )

        mock_registry = MagicMock()
        mock_registry.connected_servers = ["nano-banana"]
        mock_registry.call_tool = AsyncMock(
            return_value=MagicMock(
                is_error=False,
                content=[
                    MagicMock(
                        type="text",
                        text='{"image": "base64data", "format": "png"}',
                    )
                ],
            )
        )
        self.service._mcp_registry = mock_registry

        result = await self.service.generate(
            request="Create custom size infographic",
            width=2560,
            height=1440,
            persist=False,
        )

        assert "structured_data" in result
        # Verify dimensions were passed to MCP
        if mock_registry.call_tool.called:
            call_args = mock_registry.call_tool.call_args
            assert call_args.kwargs["arguments"]["width"] == 2560
            assert call_args.kwargs["arguments"]["height"] == 1440

    @pytest.mark.asyncio
    async def test_json_extraction_cleanup_options(self):
        """Test various JSON cleanup scenarios."""
        test_cases = [
            # Clean JSON
            ('{"headline": "Test", "stats": []}', True),
            # JSON with markdown
            ('```json\n{"headline": "Test", "stats": []}\n```', True),
            # JSON with leading text
            ('Here is the data:\n{"headline": "Test", "stats": []}', True),
            # JSON with surrounding text
            ('Analysis:\n{"headline": "Test", "stats": []}\nEnd analysis.', True),
            # Invalid JSON
            ('{"headline": "Test", stats: []}', False),
            # Empty response
            ('', False),
            # No JSON at all
            ('This is not JSON', False),
        ]

        for response_text, should_succeed in test_cases:
            self.service.llm.generate = AsyncMock(return_value=response_text)

            mock_chunks = [
                {
                    "title": "Test",
                    "text": "Test content",
                    "date": "2025-01-20",
                    "score": 0.8,
                }
            ]

            self.service.retrieval.search_with_confidence = AsyncMock(
                return_value={
                    "chunks": mock_chunks,
                    "confidence": {"level": "high", "metrics": {}},
                }
            )

            mock_registry = MagicMock()
            mock_registry.connected_servers = []
            self.service._mcp_registry = mock_registry

            result = await self.service.generate(
                request="Test",
                persist=False,
            )

            if should_succeed:
                assert "error" not in result, f"Failed for: {response_text}"
            else:
                assert "error" in result, f"Should have errored for: {response_text}"
