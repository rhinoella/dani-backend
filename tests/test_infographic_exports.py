"""
Tests for infographic schema format and export endpoints.

Tests the output_format parameter and Claude/Gemini export formats.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import uuid

from app.api.routes.infographic import (
    OutputFormat,
    ClaudeDeckExport,
    ClaudeSlide,
    GeminiVisualExport,
    GeminiVisualElement,
    ExportRequest,
    InfographicRequest,
)
from app.services.infographic_service import InfographicService, InfographicStyle


class TestOutputFormat:
    """Tests for the OutputFormat enum."""

    def test_output_format_values(self):
        """Test OutputFormat enum values."""
        assert OutputFormat.VISUAL.value == "visual"
        assert OutputFormat.SCHEMA.value == "schema"
        assert OutputFormat.BOTH.value == "both"

    def test_output_format_from_string(self):
        """Test creating OutputFormat from string."""
        assert OutputFormat("visual") == OutputFormat.VISUAL
        assert OutputFormat("schema") == OutputFormat.SCHEMA
        assert OutputFormat("both") == OutputFormat.BOTH

    def test_invalid_output_format(self):
        """Test invalid output format raises error."""
        with pytest.raises(ValueError):
            OutputFormat("invalid")


class TestInfographicRequestSchema:
    """Tests for the InfographicRequest schema."""

    def test_request_with_output_format(self):
        """Test InfographicRequest accepts output_format."""
        req = InfographicRequest(
            request="Create infographic about sales",
            output_format="schema"
        )
        assert req.output_format == "schema"

    def test_request_default_output_format(self):
        """Test default output_format is visual."""
        req = InfographicRequest(
            request="Create infographic about sales"
        )
        assert req.output_format == "visual"

    def test_request_with_all_fields(self):
        """Test InfographicRequest with all fields."""
        req = InfographicRequest(
            request="Create infographic about Q1 metrics",
            topic="Q1 2024",
            style="corporate",
            doc_type="meeting",
            width=1920,
            height=1080,
            output_format="both"
        )
        assert req.request == "Create infographic about Q1 metrics"
        assert req.style == "corporate"
        assert req.output_format == "both"


class TestClaudeDeckSchema:
    """Tests for Claude deck export schema."""

    def test_claude_slide_schema(self):
        """Test ClaudeSlide schema validation."""
        slide = ClaudeSlide(
            slide_number=1,
            title="Introduction",
            content_type="title",
            main_text="Welcome to our presentation",
            bullet_points=None,
            visual_suggestion="Full-width gradient background",
            speaker_notes="Open with enthusiasm"
        )
        assert slide.slide_number == 1
        assert slide.content_type == "title"

    def test_claude_slide_with_bullets(self):
        """Test ClaudeSlide with bullet points."""
        slide = ClaudeSlide(
            slide_number=2,
            title="Key Metrics",
            content_type="stats",
            main_text="Critical numbers",
            bullet_points=["Revenue: $1M", "Growth: 25%"],
            visual_suggestion="Stat cards layout",
            speaker_notes="Highlight the growth"
        )
        assert len(slide.bullet_points) == 2

    def test_claude_deck_schema(self):
        """Test ClaudeDeckExport schema validation."""
        deck = ClaudeDeckExport(
            deck_title="Q1 Review",
            deck_subtitle="Company Performance",
            total_slides=3,
            slides=[
                ClaudeSlide(
                    slide_number=1,
                    title="Q1 Review",
                    content_type="title",
                    main_text="Company Performance",
                    visual_suggestion="Title slide",
                    speaker_notes="Introduction"
                )
            ],
            theme_suggestion="Modern blue theme",
            source_attribution="Based on Q1 reports",
            generation_prompt="Create a professional deck"
        )
        assert deck.total_slides == 3
        assert len(deck.slides) == 1


class TestGeminiVisualSchema:
    """Tests for Gemini visual export schema."""

    def test_gemini_element_schema(self):
        """Test GeminiVisualElement schema validation."""
        element = GeminiVisualElement(
            element_type="title",
            content="Main Headline",
            position={"x": 10, "y": 5, "width": 80, "height": 10},
            style={"font_size": 48, "color": "#000000"},
            priority=1
        )
        assert element.element_type == "title"
        assert element.priority == 1

    def test_gemini_element_priority_bounds(self):
        """Test GeminiVisualElement priority bounds."""
        # Valid priority
        element = GeminiVisualElement(
            element_type="stat_card",
            content="100%",
            position={"x": 0, "y": 0, "width": 50, "height": 50},
            style={},
            priority=10
        )
        assert element.priority == 10

    def test_gemini_visual_export_schema(self):
        """Test GeminiVisualExport schema validation."""
        export = GeminiVisualExport(
            canvas_width=1024,
            canvas_height=1024,
            background_color="#ffffff",
            color_palette=["#667eea", "#764ba2"],
            elements=[
                GeminiVisualElement(
                    element_type="title",
                    content="Headline",
                    position={"x": 10, "y": 5, "width": 80, "height": 10},
                    style={"font_size": 48},
                    priority=1
                )
            ],
            layout_type="dashboard",
            generation_instructions="Create infographic"
        )
        assert export.canvas_width == 1024
        assert export.layout_type == "dashboard"


class TestExportRequest:
    """Tests for ExportRequest schema."""

    def test_export_with_infographic_id(self):
        """Test ExportRequest with existing infographic."""
        req = ExportRequest(
            infographic_id="inf-123",
            slides_count=7
        )
        assert req.infographic_id == "inf-123"
        assert req.slides_count == 7

    def test_export_with_new_request(self):
        """Test ExportRequest for new generation."""
        req = ExportRequest(
            request="Create Q1 summary",
            topic="Q1 2024",
            doc_type="meeting"
        )
        assert req.request == "Create Q1 summary"
        assert req.infographic_id is None

    def test_export_default_slides_count(self):
        """Test default slides count."""
        req = ExportRequest(request="Test")
        assert req.slides_count == 5


class TestInfographicServiceExports:
    """Tests for InfographicService export methods."""

    @pytest.fixture
    def sample_structured_data(self):
        """Sample structured infographic data."""
        return {
            "headline": "Q1 2024 Results",
            "subtitle": "Strong growth across all regions",
            "stats": [
                {"value": "$2.5M", "label": "Revenue", "icon": "💰"},
                {"value": "25%", "label": "Growth YoY", "icon": "📈"},
                {"value": "150", "label": "New Customers", "icon": "👥"},
                {"value": "98%", "label": "Satisfaction", "icon": "⭐"},
            ],
            "key_points": [
                "Revenue exceeded targets by 15%",
                "Customer acquisition cost reduced by 20%",
                "New product launch successful",
            ],
            "source_summary": "Based on Q1 2024 financial reports",
        }

    @pytest.fixture
    def sample_sources(self):
        """Sample source documents."""
        return [
            {"title": "Q1 Financial Review", "date": "2024-03-15", "score": 0.95},
            {"title": "Sales Team Update", "date": "2024-03-10", "score": 0.87},
        ]

    @pytest.mark.asyncio
    async def test_export_to_claude_deck(self, sample_structured_data, sample_sources):
        """Test Claude deck export generation."""
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            
            result = await service.export_to_claude_deck(
                structured_data=sample_structured_data,
                sources=sample_sources,
                slides_count=5,
            )
            
            assert result["deck_title"] == "Q1 2024 Results"
            assert result["total_slides"] >= 3  # At least title, stats, summary
            assert result["total_slides"] <= 5  # Up to requested count
            assert len(result["slides"]) == result["total_slides"]
            assert result["slides"][0]["content_type"] == "title"
            assert "theme_suggestion" in result
            assert "generation_prompt" in result

    @pytest.mark.asyncio
    async def test_export_to_claude_deck_slide_types(self, sample_structured_data, sample_sources):
        """Test Claude deck has correct slide types."""
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            
            result = await service.export_to_claude_deck(
                structured_data=sample_structured_data,
                sources=sample_sources,
                slides_count=5,
            )
            
            content_types = [s["content_type"] for s in result["slides"]]
            assert "title" in content_types
            assert "stats" in content_types or "key_points" in content_types

    @pytest.mark.asyncio
    async def test_export_to_gemini_visual(self, sample_structured_data):
        """Test Gemini visual export generation."""
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            
            result = await service.export_to_gemini_visual(
                structured_data=sample_structured_data,
                width=1920,
                height=1080,
            )
            
            assert result["canvas_width"] == 1920
            assert result["canvas_height"] == 1080
            assert result["background_color"] == "#ffffff"
            assert len(result["color_palette"]) > 0
            assert len(result["elements"]) > 0
            assert result["layout_type"] in ["grid", "flow", "hero", "dashboard"]

    @pytest.mark.asyncio
    async def test_export_to_gemini_visual_element_types(self, sample_structured_data):
        """Test Gemini export has required element types."""
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            
            result = await service.export_to_gemini_visual(
                structured_data=sample_structured_data,
                width=1024,
                height=1024,
            )
            
            element_types = [e["element_type"] for e in result["elements"]]
            assert "title" in element_types
            assert "stat_card" in element_types  # Should have stats
            assert "footer" in element_types

    @pytest.mark.asyncio
    async def test_gemini_stat_card_positions(self, sample_structured_data):
        """Test Gemini stat cards have valid positions."""
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            
            result = await service.export_to_gemini_visual(
                structured_data=sample_structured_data,
                width=1024,
                height=1024,
            )
            
            stat_cards = [e for e in result["elements"] if e["element_type"] == "stat_card"]
            
            for card in stat_cards:
                pos = card["position"]
                assert 0 <= pos["x"] <= 100
                assert 0 <= pos["y"] <= 100
                assert pos["width"] > 0
                assert pos["height"] > 0


class TestGenerateSchemaOnly:
    """Tests for schema-only generation."""

    @pytest.mark.asyncio
    async def test_generate_schema_only_returns_no_image(self):
        """Test schema-only mode returns no image data."""
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            service.retrieval = MagicMock()
            service.retrieval.search_with_confidence = AsyncMock(return_value={
                "chunks": [{"title": "Test", "text": "Content", "date": "2024-01-01"}],
                "confidence": {"score": 0.9, "level": "high"},
            })
            service._extract_structured_data = AsyncMock(return_value={
                "headline": "Test",
                "stats": [{"value": "100", "label": "Metric"}],
            })
            
            result = await service.generate_schema_only(
                request="Test request",
            )
            
            assert result["image"] is None
            assert result["image_url"] is None
            assert result["structured_data"]["headline"] == "Test"
            assert result["metadata"]["output_format"] == "schema"

    @pytest.mark.asyncio
    async def test_generate_schema_only_timing(self):
        """Test schema-only mode has correct timing fields."""
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            service.retrieval = MagicMock()
            service.retrieval.search_with_confidence = AsyncMock(return_value={
                "chunks": [{"title": "Test", "text": "Content"}],
                "confidence": {"score": 0.9, "level": "high"},
            })
            service._extract_structured_data = AsyncMock(return_value={
                "headline": "Test",
                "stats": [],
            })
            
            result = await service.generate_schema_only(request="Test")
            
            assert "timing" in result
            assert result["timing"]["image_ms"] == 0  # No image generation
            assert result["timing"]["retrieval_ms"] >= 0
            assert result["timing"]["extraction_ms"] >= 0


class TestContentTypes:
    """Test content type literals."""

    def test_slide_content_types(self):
        """Test all slide content types are valid."""
        valid_types = ["title", "stats", "key_points", "comparison", "summary"]
        
        for ct in valid_types:
            slide = ClaudeSlide(
                slide_number=1,
                title="Test",
                content_type=ct,
                main_text="Test",
                visual_suggestion="Test",
                speaker_notes="Test"
            )
            assert slide.content_type == ct

    def test_element_types(self):
        """Test all Gemini element types are valid."""
        valid_types = ["title", "subtitle", "stat_card", "icon", "chart", "bullet_list", "divider", "footer"]
        
        for et in valid_types:
            element = GeminiVisualElement(
                element_type=et,
                content="Test",
                position={"x": 0, "y": 0, "width": 10, "height": 10},
                style={},
                priority=1
            )
            assert element.element_type == et

    def test_layout_types(self):
        """Test all Gemini layout types are valid."""
        valid_layouts = ["grid", "flow", "hero", "dashboard"]
        
        for layout in valid_layouts:
            export = GeminiVisualExport(
                canvas_width=1024,
                canvas_height=1024,
                background_color="#fff",
                color_palette=[],
                elements=[],
                layout_type=layout,
                generation_instructions="Test"
            )
            assert export.layout_type == layout
