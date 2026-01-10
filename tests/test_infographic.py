"""Tests for InfographicService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from app.services.infographic_service import (
    InfographicService,
    InfographicStyle,
    get_infographic_service,
    STYLE_PROMPTS,
)


class TestInfographicService:
    """Tests for InfographicService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = InfographicService()
        
    def test_styles_enum(self):
        """Should have all expected styles."""
        assert InfographicStyle.MODERN.value == "modern"
        assert InfographicStyle.CORPORATE.value == "corporate"
        assert InfographicStyle.MINIMAL.value == "minimal"
        assert InfographicStyle.VIBRANT.value == "vibrant"
        assert InfographicStyle.DARK.value == "dark"

    def test_all_styles_have_prompts(self):
        """Every style should have a corresponding prompt."""
        for style in InfographicStyle:
            assert style in STYLE_PROMPTS
            assert len(STYLE_PROMPTS[style]) > 50  # Meaningful prompt

    def test_get_styles(self):
        """Should return list of styles with descriptions."""
        styles = self.service.get_styles()
        
        assert len(styles) == len(InfographicStyle)
        for s in styles:
            assert "style" in s
            assert "description" in s
            assert s["style"] in [style.value for style in InfographicStyle]

    @pytest.mark.asyncio
    async def test_generate_no_context(self):
        """Should return error when no context found."""
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "none", "metrics": {}},
        })

        result = await self.service.generate(
            request="Create infographic about mobile strategy",
        )

        assert "error" in result
        assert "No relevant context" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_extraction_success(self):
        """Should extract structured data from context."""
        mock_chunks = [
            {
                "title": "Q1 Strategy Meeting",
                "text": "Mobile app launching March 2025. Budget $500K approved.",
                "date": "2025-01-03",
                "score": 0.9,
            }
        ]
        
        mock_structured = {
            "headline": "Mobile Strategy 2025",
            "subtitle": "Key milestones and metrics",
            "stats": [
                {"value": "March 2025", "label": "Launch Date", "icon": "📱"},
                {"value": "$500K", "label": "Budget", "icon": "💰"},
            ],
            "key_points": ["Mobile-first approach", "Q1 priority"],
        }
        
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": mock_chunks,
            "confidence": {"level": "high", "metrics": {}},
        })
        
        self.service.llm.generate = AsyncMock(return_value=json.dumps(mock_structured))
        
        # Mock MCP registry as not connected
        mock_registry = MagicMock()
        mock_registry.connected_servers = []
        self.service._mcp_registry = mock_registry

        result = await self.service.generate(
            request="Create infographic about mobile strategy",
        )

        # Should have structured data even if image fails
        assert "structured_data" in result
        assert result["structured_data"]["headline"] == "Mobile Strategy 2025"
        assert len(result["structured_data"]["stats"]) == 2
        assert "image_error" in result  # Image generation failed (no MCP)

    @pytest.mark.asyncio
    async def test_generate_with_mcp_image(self):
        """Should generate image via MCP when available."""
        mock_chunks = [
            {
                "title": "Q1 Meeting",
                "text": "Revenue up 30%. Team expanding to 50 engineers.",
                "date": "2025-01-03",
                "score": 0.85,
            }
        ]
        
        mock_structured = {
            "headline": "Q1 Performance",
            "stats": [
                {"value": "30%", "label": "Revenue Growth", "icon": "📈"},
                {"value": "50", "label": "Engineers", "icon": "👥"},
            ],
        }
        
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": mock_chunks,
            "confidence": {"level": "high", "metrics": {}},
        })
        
        self.service.llm.generate = AsyncMock(return_value=json.dumps(mock_structured))
        
        # Mock MCP registry with connected nano-banana
        # MCPToolResult uses is_error flag and content is a list
        mock_registry = MagicMock()
        mock_registry.connected_servers = ["nano-banana"]
        mock_registry.call_tool = AsyncMock(return_value=MagicMock(
            is_error=False,
            content=[MagicMock(type="text", text='{"image": "base64_image_data_here", "format": "png"}')],
        ))
        self.service._mcp_registry = mock_registry

        result = await self.service.generate(
            request="Create Q1 performance infographic",
            style=InfographicStyle.CORPORATE,
        )

        assert "structured_data" in result
        # Image may be extracted differently based on content format
        assert result["metadata"]["style"] == "corporate"
        
        # Verify MCP was called
        mock_registry.call_tool.assert_called_once()
        call_args = mock_registry.call_tool.call_args
        assert call_args.kwargs["server_name"] == "nano-banana"
        assert call_args.kwargs["tool_name"] == "generate_image"

    @pytest.mark.asyncio
    async def test_generate_with_doc_type_filter(self):
        """Should pass doc_type filter to retrieval."""
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "none", "metrics": {}},
        })

        await self.service.generate(
            request="Meeting infographic",
            doc_type="meeting",
        )

        # Check that filter was passed
        call_args = self.service.retrieval.search_with_confidence.call_args
        assert call_args.kwargs["metadata_filter"] is not None
        assert call_args.kwargs["metadata_filter"].doc_type == "meeting"

    @pytest.mark.asyncio
    async def test_extract_structured_data_json_cleanup(self):
        """Should clean markdown code blocks from JSON response."""
        context = "Meeting about mobile launch in March."
        request = "Create infographic"
        
        # LLM returns JSON wrapped in markdown
        json_with_markdown = """```json
{
    "headline": "Mobile Launch",
    "stats": [{"value": "March", "label": "Launch", "icon": "📱"}]
}
```"""
        
        self.service.llm.generate = AsyncMock(return_value=json_with_markdown)
        
        result = await self.service._extract_structured_data(context, request)
        
        assert "error" not in result
        assert result["headline"] == "Mobile Launch"

    @pytest.mark.asyncio
    async def test_extract_structured_data_missing_field(self):
        """Should return error if required fields missing."""
        context = "Some context"
        request = "Create infographic"
        
        # Missing 'stats' field
        incomplete_json = '{"headline": "Test"}'
        self.service.llm.generate = AsyncMock(return_value=incomplete_json)
        
        result = await self.service._extract_structured_data(context, request)
        
        assert "error" in result
        assert "stats" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_structured_data_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        context = "Some context"
        request = "Create infographic"
        
        self.service.llm.generate = AsyncMock(return_value="This is not JSON at all")
        
        result = await self.service._extract_structured_data(context, request)
        
        assert "error" in result
        assert "parse" in result["error"].lower()

    def test_build_image_prompt(self):
        """Should build comprehensive image prompt."""
        structured_data = {
            "headline": "Q1 Results",
            "subtitle": "Strong performance",
            "stats": [
                {"value": "30%", "label": "Growth", "icon": "📈"},
                {"value": "$1M", "label": "Revenue", "icon": "💰"},
            ],
            "key_points": ["Record quarter", "Team expansion"],
        }
        
        prompt = self.service._build_image_prompt(
            structured_data, 
            InfographicStyle.VIBRANT
        )
        
        assert "Q1 Results" in prompt
        assert "Strong performance" in prompt
        assert "30%" in prompt
        assert "Growth" in prompt
        assert "Record quarter" in prompt
        assert "vibrant" in prompt.lower() or "colorful" in prompt.lower()
        assert "infographic" in prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_custom_dimensions(self):
        """Should pass custom width/height to image generation."""
        mock_chunks = [{"title": "Test", "text": "Content", "date": "2025-01-01", "score": 0.8}]
        
        self.service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": mock_chunks,
            "confidence": {"level": "medium", "metrics": {}},
        })
        
        self.service.llm.generate = AsyncMock(return_value='{"headline": "Test", "stats": []}')
        
        mock_registry = MagicMock()
        mock_registry.connected_servers = ["nano-banana"]
        mock_registry.call_tool = AsyncMock(return_value=MagicMock(
            is_error=False,
            content=[MagicMock(type="text", text='{"image": "data", "format": "png"}')],
        ))
        self.service._mcp_registry = mock_registry

        await self.service.generate(
            request="Test",
            width=1920,
            height=1080,
        )

        call_args = mock_registry.call_tool.call_args
        assert call_args.kwargs["arguments"]["width"] == 1920
        assert call_args.kwargs["arguments"]["height"] == 1080


class TestInfographicSingleton:
    """Tests for singleton pattern."""

    def test_get_infographic_service_returns_same_instance(self):
        """Should return same instance on repeated calls."""
        # Reset singleton
        import app.services.infographic_service as module
        module._infographic_service = None
        
        service1 = get_infographic_service()
        service2 = get_infographic_service()
        
        assert service1 is service2

    def test_singleton_is_infographic_service(self):
        """Singleton should be InfographicService instance."""
        import app.services.infographic_service as module
        module._infographic_service = None
        
        service = get_infographic_service()
        assert isinstance(service, InfographicService)


class TestInfographicImageGeneration:
    """Tests for image generation via MCP."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = InfographicService()

    @pytest.mark.asyncio
    async def test_generate_image_mcp_not_connected(self):
        """Should return error when MCP not connected."""
        mock_registry = MagicMock()
        mock_registry.connected_servers = []
        self.service._mcp_registry = mock_registry

        result = await self.service._generate_image(
            {"headline": "Test", "stats": []},
            InfographicStyle.MODERN,
            1024, 1024,
        )

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_image_mcp_failure(self):
        """Should handle MCP tool failure gracefully."""
        from app.mcp.schemas import MCPTextContent
        
        mock_registry = MagicMock()
        mock_registry.connected_servers = ["nano-banana"]
        mock_registry.call_tool = AsyncMock(return_value=MagicMock(
            is_error=True,
            content=[MCPTextContent(type="text", text="API quota exceeded")],
        ))
        self.service._mcp_registry = mock_registry

        result = await self.service._generate_image(
            {"headline": "Test", "stats": []},
            InfographicStyle.MODERN,
            1024, 1024,
        )

        assert "error" in result
        assert "quota" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_generate_image_url_response(self):
        """Should handle URL response from MCP."""
        from app.mcp.schemas import MCPTextContent
        
        mock_registry = MagicMock()
        mock_registry.connected_servers = ["nano-banana"]
        mock_registry.call_tool = AsyncMock(return_value=MagicMock(
            is_error=False,
            content=[MCPTextContent(type="text", text="https://example.com/image.png")],
        ))
        self.service._mcp_registry = mock_registry

        result = await self.service._generate_image(
            {"headline": "Test", "stats": []},
            InfographicStyle.MODERN,
            1024, 1024,
        )

        assert result["url"] == "https://example.com/image.png"
        assert result["format"] == "png"

    @pytest.mark.asyncio
    async def test_generate_image_dict_response(self):
        """Should handle JSON dict response with image data."""
        from app.mcp.schemas import MCPTextContent
        
        mock_registry = MagicMock()
        mock_registry.connected_servers = ["nano-banana"]
        mock_registry.call_tool = AsyncMock(return_value=MagicMock(
            is_error=False,
            content=[MCPTextContent(
                type="text", 
                text='{"data": "base64_encoded_image", "url": "https://hosted.com/img.png", "format": "png"}'
            )],
        ))
        self.service._mcp_registry = mock_registry

        result = await self.service._generate_image(
            {"headline": "Test", "stats": []},
            InfographicStyle.DARK,
            1024, 1024,
        )

        assert result["image"] == "base64_encoded_image"
        assert result["url"] == "https://hosted.com/img.png"

    @pytest.mark.asyncio
    async def test_generate_image_binary_response(self):
        """Should handle binary image response from MCP."""
        from app.mcp.schemas import MCPImageContent
        
        mock_registry = MagicMock()
        mock_registry.connected_servers = ["nano-banana"]
        mock_registry.call_tool = AsyncMock(return_value=MagicMock(
            is_error=False,
            content=[MCPImageContent(type="image", data="base64_binary_data", mimeType="image/png")],
        ))
        self.service._mcp_registry = mock_registry

        result = await self.service._generate_image(
            {"headline": "Test", "stats": []},
            InfographicStyle.VIBRANT,
            1024, 1024,
        )

        assert result["image"] == "base64_binary_data"
        assert result["format"] == "png"
