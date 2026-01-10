"""
Tests for Infographic Spec Schema Format.

Tests the PROJECT PLAN spec schema conversion and generation
for infographic data.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.infographic_service import (
    InfographicService,
    get_infographic_service,
)


class TestToSpecSchema:
    """Test to_spec_schema conversion method."""
    
    def test_basic_conversion(self):
        """Test basic structured data to spec schema conversion."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Q1 Results Summary",
            "subtitle": "Strong performance across all metrics",
            "stats": [
                {"value": "25%", "label": "Revenue Growth", "icon": "📈"},
                {"value": "1.2M", "label": "Active Users", "icon": "👥"},
            ],
            "key_points": [
                "Mobile adoption increased significantly",
                "Enterprise deals exceeded targets",
            ],
            "source_summary": "From Q1 All-Hands Meeting",
        }
        
        spec = service.to_spec_schema(structured_data)
        
        assert "title" in spec
        assert spec["title"] == "Q1 Results Summary"
        assert "sections" in spec
        assert "recommended_visuals" in spec
    
    def test_sections_structure(self):
        """Test that sections have correct header/bullets structure."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Test Title",
            "subtitle": "Test subtitle",
            "stats": [
                {"value": "100", "label": "Test stat", "icon": "📊"},
            ],
            "key_points": ["Point 1", "Point 2"],
            "source_summary": "Test source",
        }
        
        spec = service.to_spec_schema(structured_data)
        
        # All sections should have header and bullets
        for section in spec["sections"]:
            assert "header" in section
            assert "bullets" in section
            assert isinstance(section["bullets"], list)
    
    def test_overview_section_from_subtitle(self):
        """Test that subtitle creates an Overview section."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Title",
            "subtitle": "This is the overview context",
            "stats": [],
            "key_points": [],
        }
        
        spec = service.to_spec_schema(structured_data)
        
        overview = next(
            (s for s in spec["sections"] if s["header"] == "Overview"),
            None
        )
        assert overview is not None
        assert "This is the overview context" in overview["bullets"]
    
    def test_key_metrics_section_from_stats(self):
        """Test that stats create a Key Metrics section."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Title",
            "stats": [
                {"value": "50%", "label": "Growth Rate", "icon": "📈"},
                {"value": "$1M", "label": "Revenue", "icon": "💰"},
            ],
            "key_points": [],
        }
        
        spec = service.to_spec_schema(structured_data)
        
        metrics = next(
            (s for s in spec["sections"] if s["header"] == "Key Metrics"),
            None
        )
        assert metrics is not None
        assert len(metrics["bullets"]) == 2
        assert any("50%" in b for b in metrics["bullets"])
        assert any("$1M" in b for b in metrics["bullets"])
    
    def test_key_insights_section_from_key_points(self):
        """Test that key_points create a Key Insights section."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Title",
            "stats": [],
            "key_points": [
                "First key insight",
                "Second key insight",
            ],
        }
        
        spec = service.to_spec_schema(structured_data)
        
        insights = next(
            (s for s in spec["sections"] if s["header"] == "Key Insights"),
            None
        )
        assert insights is not None
        assert "First key insight" in insights["bullets"]
        assert "Second key insight" in insights["bullets"]
    
    def test_source_section_from_summary(self):
        """Test that source_summary creates a Source section."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Title",
            "stats": [],
            "key_points": [],
            "source_summary": "From Product Meeting on 2024-01-15",
        }
        
        spec = service.to_spec_schema(structured_data)
        
        source = next(
            (s for s in spec["sections"] if s["header"] == "Source"),
            None
        )
        assert source is not None
        assert "From Product Meeting on 2024-01-15" in source["bullets"]
    
    def test_empty_data_handling(self):
        """Test handling of minimal/empty structured data."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Minimal Title",
        }
        
        spec = service.to_spec_schema(structured_data)
        
        assert spec["title"] == "Minimal Title"
        assert isinstance(spec["sections"], list)
        assert "recommended_visuals" in spec


class TestRecommendedVisuals:
    """Test recommended_visuals generation."""
    
    def test_dashboard_recommendation_for_many_stats(self):
        """Test dashboard layout recommended for 4+ stats."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Dashboard Test",
            "stats": [
                {"value": "1", "label": "Stat 1", "icon": "📊"},
                {"value": "2", "label": "Stat 2", "icon": "📊"},
                {"value": "3", "label": "Stat 3", "icon": "📊"},
                {"value": "4", "label": "Stat 4", "icon": "📊"},
            ],
            "key_points": [],
        }
        
        spec = service.to_spec_schema(structured_data)
        
        assert "Dashboard" in spec["recommended_visuals"] or "dashboard" in spec["recommended_visuals"].lower()
    
    def test_percentage_chart_recommendation(self):
        """Test pie/progress chart recommended for percentage stats."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Percentage Test",
            "stats": [
                {"value": "75%", "label": "Completion", "icon": "📊"},
            ],
            "key_points": [],
        }
        
        spec = service.to_spec_schema(structured_data)
        
        # Should mention pie chart or progress bars
        visuals_lower = spec["recommended_visuals"].lower()
        assert "pie" in visuals_lower or "progress" in visuals_lower or "percentage" in visuals_lower
    
    def test_currency_chart_recommendation(self):
        """Test bar chart recommended for currency stats."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Financial Test",
            "stats": [
                {"value": "$500K", "label": "Revenue", "icon": "💰"},
            ],
            "key_points": [],
        }
        
        spec = service.to_spec_schema(structured_data)
        
        visuals_lower = spec["recommended_visuals"].lower()
        assert "bar" in visuals_lower or "financial" in visuals_lower or "chart" in visuals_lower
    
    def test_bullet_recommendation_for_many_points(self):
        """Test bullet list recommended for multiple key points."""
        service = InfographicService()
        
        structured_data = {
            "headline": "Points Test",
            "stats": [],
            "key_points": ["Point 1", "Point 2", "Point 3", "Point 4"],
        }
        
        spec = service.to_spec_schema(structured_data)
        
        visuals_lower = spec["recommended_visuals"].lower()
        assert "bullet" in visuals_lower or "list" in visuals_lower or "card" in visuals_lower


class TestGenerateSpecSchema:
    """Test generate_spec_schema method."""
    
    @pytest.mark.asyncio
    async def test_generate_returns_spec_format(self):
        """Test that generate_spec_schema returns spec format."""
        service = InfographicService()
        
        # Mock retrieval service
        service.retrieval = MagicMock()
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [
                {"title": "Test Meeting", "text": "Content here", "date": "2024-01-15", "score": 0.9}
            ],
            "confidence": {"level": "high", "score": 0.9},
        })
        
        # Mock extraction
        with patch.object(service, '_extract_structured_data') as mock_extract:
            mock_extract.return_value = {
                "headline": "Test Infographic",
                "subtitle": "Test subtitle",
                "stats": [{"value": "100", "label": "Metric", "icon": "📊"}],
                "key_points": ["Point 1"],
                "source_summary": "Test source",
            }
            
            result = await service.generate_spec_schema(
                request="Test infographic request",
            )
        
        assert "spec_schema" in result
        assert "title" in result["spec_schema"]
        assert "sections" in result["spec_schema"]
        assert "recommended_visuals" in result["spec_schema"]
        assert "raw_structured_data" in result
    
    @pytest.mark.asyncio
    async def test_generate_handles_no_context(self):
        """Test that generate_spec_schema handles no context gracefully."""
        service = InfographicService()
        
        # Mock retrieval to return no chunks
        service.retrieval = MagicMock()
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [],
            "confidence": {"level": "low", "score": 0.0},
        })
        
        result = await service.generate_spec_schema(
            request="Test request with no data",
        )
        
        assert "error" in result
        assert "No relevant context" in result["error"]
    
    @pytest.mark.asyncio
    async def test_generate_includes_metadata(self):
        """Test that generate_spec_schema includes proper metadata."""
        service = InfographicService()
        
        # Mock retrieval
        service.retrieval = MagicMock()
        service.retrieval.search_with_confidence = AsyncMock(return_value={
            "chunks": [
                {"title": "Meeting", "text": "Text", "date": None, "score": 0.8}
            ],
            "confidence": {"level": "medium", "score": 0.75},
        })
        
        # Mock extraction
        with patch.object(service, '_extract_structured_data') as mock_extract:
            mock_extract.return_value = {
                "headline": "Test",
                "stats": [],
                "key_points": [],
            }
            
            result = await service.generate_spec_schema(
                request="Test request",
            )
        
        assert "timing" in result
        assert "retrieval_ms" in result["timing"]
        assert "extraction_ms" in result["timing"]
        assert "metadata" in result
        assert result["metadata"]["output_format"] == "spec_schema"


class TestSpecSchemaAPIEndpoint:
    """Test the spec schema API endpoints."""
    
    @pytest.mark.asyncio
    async def test_convert_endpoint_schema_structure(self):
        """Test convert endpoint produces valid spec schema structure."""
        from app.api.routes.infographic import SpecSchema, SpecSchemaSection
        
        # Test that schema models work correctly
        section = SpecSchemaSection(
            header="Test Header",
            bullets=["Bullet 1", "Bullet 2"]
        )
        
        schema = SpecSchema(
            title="Test Title",
            sections=[section],
            recommended_visuals="Test visuals"
        )
        
        assert schema.title == "Test Title"
        assert len(schema.sections) == 1
        assert schema.sections[0].header == "Test Header"
        assert schema.recommended_visuals == "Test visuals"
    
    @pytest.mark.asyncio
    async def test_spec_schema_response_structure(self):
        """Test SpecSchemaResponse model structure."""
        from app.api.routes.infographic import (
            SpecSchemaResponse,
            SpecSchema,
            SpecSchemaSection,
            StructuredData,
            InfographicStat,
        )
        
        spec_schema = SpecSchema(
            title="Test",
            sections=[SpecSchemaSection(header="H", bullets=["B"])],
            recommended_visuals="Test"
        )
        
        structured = StructuredData(
            headline="Test",
            stats=[InfographicStat(value="1", label="Test")],
        )
        
        response = SpecSchemaResponse(
            spec_schema=spec_schema,
            raw_structured_data=structured,
            sources=[],
            confidence={"level": "high"},
            timing={"total_ms": 100},
            metadata={"output_format": "spec_schema"},
        )
        
        assert response.spec_schema.title == "Test"
        assert response.raw_structured_data.headline == "Test"


class TestFullIntegration:
    """Full integration tests for spec schema flow."""
    
    def test_full_conversion_flow(self):
        """Test complete conversion from internal format to spec schema."""
        service = InfographicService()
        
        # Realistic structured data
        structured_data = {
            "headline": "Q1 2024 Mobile Strategy Review",
            "subtitle": "Key findings from the quarterly planning meeting",
            "stats": [
                {"value": "2.5M", "label": "Monthly Active Users", "icon": "👥"},
                {"value": "45%", "label": "YoY Growth", "icon": "📈"},
                {"value": "$3.2M", "label": "App Revenue", "icon": "💰"},
                {"value": "4.8★", "label": "App Store Rating", "icon": "⭐"},
            ],
            "key_points": [
                "Push notification engagement up 32%",
                "New onboarding flow reducing churn by 18%",
                "Android performance now matches iOS",
                "Premium tier adoption exceeded projections",
            ],
            "source_summary": "Q1 Strategy Meeting - January 15, 2024",
        }
        
        spec = service.to_spec_schema(structured_data)
        
        # Validate complete spec structure
        assert spec["title"] == "Q1 2024 Mobile Strategy Review"
        
        # Should have multiple sections
        assert len(spec["sections"]) >= 3
        
        # Find and validate each section type
        headers = [s["header"] for s in spec["sections"]]
        assert "Overview" in headers
        assert "Key Metrics" in headers
        assert "Key Insights" in headers
        assert "Source" in headers
        
        # Key Metrics should have all 4 stats
        metrics = next(s for s in spec["sections"] if s["header"] == "Key Metrics")
        assert len(metrics["bullets"]) == 4
        
        # Key Insights should have all 4 points
        insights = next(s for s in spec["sections"] if s["header"] == "Key Insights")
        assert len(insights["bullets"]) == 4
        
        # Recommended visuals should mention dashboard (4+ stats) and percentages
        visuals = spec["recommended_visuals"].lower()
        assert "dashboard" in visuals or "metric" in visuals
