"""
Tests for Email Style Service.

Tests the email corpus indexing and style pattern extraction
for the ghostwriter email style learning feature.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.email_style_service import (
    EmailStyleService,
    EmailTone,
)


class TestEmailToneEnum:
    """Test EmailTone enum."""
    
    def test_tone_values(self):
        """Test that all expected tones are defined."""
        assert EmailTone.FORMAL.value == "formal"
        assert EmailTone.INFORMAL.value == "informal"
        assert EmailTone.URGENT.value == "urgent"
        assert EmailTone.FRIENDLY.value == "friendly"
        assert EmailTone.DIRECT.value == "direct"


class TestEmailStyleServiceInit:
    """Test EmailStyleService initialization."""
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_service_initializes(self, mock_embed, mock_qdrant):
        """Test service initializes with dependencies."""
        service = EmailStyleService()
        
        assert service.embedder is not None
        assert service.store is not None


class TestPatternExtraction:
    """Test pattern extraction methods."""
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_extract_greeting_hi(self, mock_embed, mock_qdrant):
        """Test extraction of 'Hi' greeting."""
        service = EmailStyleService()
        
        body = "Hi John,\n\nHope you're doing well.\n\nBest,\nBunmi"
        patterns = service._extract_patterns(body)
        
        assert "greeting" in patterns
        assert "Hi" in patterns["greeting"]
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_extract_greeting_dear(self, mock_embed, mock_qdrant):
        """Test extraction of 'Dear' greeting."""
        service = EmailStyleService()
        
        body = "Dear Mr. Smith,\n\nI am writing to follow up.\n\nRegards,\nBunmi"
        patterns = service._extract_patterns(body)
        
        assert "greeting" in patterns
        assert "Dear" in patterns["greeting"]
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_extract_sign_off_best(self, mock_embed, mock_qdrant):
        """Test extraction of 'Best' sign-off."""
        service = EmailStyleService()
        
        body = "Hi team,\n\nPlease review the doc.\n\nBest,\nBunmi"
        patterns = service._extract_patterns(body)
        
        assert "sign_off" in patterns
        assert "Best" in patterns["sign_off"]
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_extract_sign_off_regards(self, mock_embed, mock_qdrant):
        """Test extraction of 'Regards' sign-off."""
        service = EmailStyleService()
        
        body = "Hi team,\n\nPlease review.\n\nRegards,\nBunmi"
        patterns = service._extract_patterns(body)
        
        assert "sign_off" in patterns
        assert "Regards" in patterns["sign_off"]
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_extract_has_cta(self, mock_embed, mock_qdrant):
        """Test extraction of call-to-action patterns."""
        service = EmailStyleService()
        
        body = "Hi team,\n\nLet me know if you have questions.\n\nBest,\nBunmi"
        patterns = service._extract_patterns(body)
        
        assert "has_cta" in patterns
        assert patterns["has_cta"] is True
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_extract_paragraph_count(self, mock_embed, mock_qdrant):
        """Test paragraph count extraction."""
        service = EmailStyleService()
        
        body = "Hi team,\n\nFirst paragraph here.\n\nSecond paragraph.\n\nBest,\nBunmi"
        patterns = service._extract_patterns(body)
        
        assert "paragraph_count" in patterns
        assert patterns["paragraph_count"] >= 2
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_extract_avg_sentence_length(self, mock_embed, mock_qdrant):
        """Test average sentence length extraction."""
        service = EmailStyleService()
        
        body = "Hi team,\n\nThis is sentence one. Here is sentence two.\n\nBest,\nBunmi"
        patterns = service._extract_patterns(body)
        
        assert "avg_sentence_length" in patterns
        assert patterns["avg_sentence_length"] > 0


class TestToneDetection:
    """Test tone detection methods."""
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_detect_formal_tone(self, mock_embed, mock_qdrant):
        """Test formal tone detection."""
        service = EmailStyleService()
        
        body = "Dear Mr. Johnson,\n\nI am pleased to inform you that your request has been approved. Please find the attached documentation for your review.\n\nSincerely,\nBunmi"
        tone = service._detect_tone(body)
        
        assert tone == EmailTone.FORMAL
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_detect_friendly_tone_with_exclamation(self, mock_embed, mock_qdrant):
        """Test friendly tone detection with exclamation."""
        service = EmailStyleService()
        
        body = "Hey! Just wanted to check in. Had some cool ideas about the project!\n\nCheers,\nBunmi"
        tone = service._detect_tone(body)
        
        assert tone == EmailTone.FRIENDLY
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_detect_urgent_tone(self, mock_embed, mock_qdrant):
        """Test urgent tone detection."""
        service = EmailStyleService()
        
        body = "URGENT: Need this ASAP! Please respond immediately. This is critical!\n\nThanks,\nBunmi"
        tone = service._detect_tone(body)
        
        assert tone == EmailTone.URGENT
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_detect_friendly_tone_hope_youre_well(self, mock_embed, mock_qdrant):
        """Test friendly tone detection with 'hope you're well'."""
        service = EmailStyleService()
        
        body = "Hi Sarah,\n\nHope you're well! I wanted to share some updates.\n\nBest,\nBunmi"
        tone = service._detect_tone(body)
        
        assert tone == EmailTone.FRIENDLY
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_detect_direct_tone(self, mock_embed, mock_qdrant):
        """Test direct tone detection (default)."""
        service = EmailStyleService()
        
        # Plain email without strong tone indicators
        body = "John,\n\nHere is the update.\n\nBunmi"
        tone = service._detect_tone(body)
        
        assert tone == EmailTone.DIRECT


class TestBuildStyleContext:
    """Test style context building."""
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_build_context_with_patterns(self, mock_embed, mock_qdrant):
        """Test building context with patterns."""
        service = EmailStyleService()
        
        patterns = {
            "common_greetings": [("Hi [Name]", 5), ("Hello", 3)],
            "common_sign_offs": [("Best,\nBunmi", 4), ("Thanks,\nBunmi", 2)],
            "avg_word_count": 150,
            "avg_paragraphs": 3,
        }
        similar_emails = []
        
        context = service.build_style_context(similar_emails, patterns)
        
        assert "BUNMI'S EMAIL STYLE PATTERNS" in context
        assert "Hi [Name]" in context
        assert "Best" in context
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_build_context_with_similar_emails(self, mock_embed, mock_qdrant):
        """Test building context with similar email examples."""
        service = EmailStyleService()
        
        patterns = {
            "common_greetings": [("Hi", 5)],
            "common_sign_offs": [("Best", 4)],
        }
        similar_emails = [
            {"text": "Hi team,\n\nThis is a sample email.\n\nBest,\nBunmi"},
        ]
        
        context = service.build_style_context(similar_emails, patterns)
        
        assert "REFERENCE EMAILS FROM BUNMI" in context
        assert "Example 1" in context
        assert "sample email" in context


class TestGetDefaultPatterns:
    """Test default patterns."""
    
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    def test_default_patterns_structure(self, mock_embed, mock_qdrant):
        """Test that default patterns have expected structure."""
        service = EmailStyleService()
        
        patterns = service._get_default_patterns()
        
        assert "common_greetings" in patterns
        assert "common_sign_offs" in patterns
        assert "avg_word_count" in patterns
        assert "avg_paragraphs" in patterns


class TestIndexEmail:
    """Test email indexing."""
    
    @pytest.mark.asyncio
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    async def test_index_email_success(self, mock_embed_cls, mock_qdrant_cls):
        """Test successful email indexing."""
        # Setup mocks
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.client = MagicMock()
        mock_qdrant.client.collection_exists = MagicMock(return_value=True)
        mock_qdrant.upsert = MagicMock()
        mock_qdrant.ensure_collection = MagicMock()
        
        mock_embed = MagicMock()
        mock_embed_cls.return_value = mock_embed
        mock_embed.embed_document = AsyncMock(return_value=[0.1] * 1536)
        
        service = EmailStyleService()
        service.store = mock_qdrant
        service.embedder = mock_embed
        service._initialized = True  # Skip ensure_collection
        
        result = await service.index_email(
            email_content="Hi team,\n\nHere's the Q1 update.\n\nBest,\nBunmi",
            subject="Q1 Planning Update",
            recipient_type="colleague",
        )
        
        assert "email_id" in result
        assert "patterns" in result
        assert result["status"] == "indexed"
    
    @pytest.mark.asyncio
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    async def test_index_email_extracts_patterns(self, mock_embed_cls, mock_qdrant_cls):
        """Test that indexing extracts patterns."""
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.client = MagicMock()
        mock_qdrant.client.collection_exists = MagicMock(return_value=True)
        mock_qdrant.upsert = MagicMock()
        mock_qdrant.ensure_collection = MagicMock()
        
        mock_embed = MagicMock()
        mock_embed_cls.return_value = mock_embed
        mock_embed.embed_document = AsyncMock(return_value=[0.1] * 1536)
        
        service = EmailStyleService()
        service.store = mock_qdrant
        service.embedder = mock_embed
        service._initialized = True  # Skip ensure_collection
        
        result = await service.index_email(
            email_content="Hi John,\n\nJust following up.\n\nLet me know your thoughts.\n\nBest regards,\nBunmi",
            subject="Follow Up",
        )
        
        patterns = result.get("patterns", {})
        assert "greeting" in patterns
        assert "sign_off" in patterns
        assert "has_cta" in patterns


class TestGetSimilarEmails:
    """Test similar email retrieval."""
    
    @pytest.mark.asyncio
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    async def test_get_similar_emails_empty(self, mock_embed_cls, mock_qdrant_cls):
        """Test retrieval when no similar emails exist."""
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.client = MagicMock()
        mock_qdrant.client.collection_exists = MagicMock(return_value=True)
        mock_qdrant.search = AsyncMock(return_value=[])  # Async mock
        mock_qdrant.ensure_collection = MagicMock()
        
        mock_embed = MagicMock()
        mock_embed_cls.return_value = mock_embed
        mock_embed.embed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_embed.embed_document = AsyncMock(return_value=[0.1] * 1536)
        
        service = EmailStyleService()
        service.store = mock_qdrant
        service.embedder = mock_embed
        service._initialized = True  # Skip ensure_collection
        
        result = await service.get_similar_emails("project update")
        
        assert result == []
    
    @pytest.mark.asyncio
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    async def test_get_similar_emails_returns_matches(self, mock_embed_cls, mock_qdrant_cls):
        """Test retrieval returns similar emails."""
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.client = MagicMock()
        mock_qdrant.client.collection_exists = MagicMock(return_value=True)
        mock_qdrant.ensure_collection = MagicMock()
        
        # Mock search result as ScoredPoint-like object
        mock_hit = MagicMock()
        mock_hit.payload = {
            "text": "Hi team, here's the update.",
            "subject": "Q1 Update",
            "recipient_type": "colleague",
            "tone": "direct",
            "greeting": "Hi team,",
            "sign_off": "Best,\nBunmi",
        }
        mock_hit.score = 0.92
        mock_qdrant.search = AsyncMock(return_value=[mock_hit])
        
        mock_embed = MagicMock()
        mock_embed_cls.return_value = mock_embed
        mock_embed.embed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_embed.embed_document = AsyncMock(return_value=[0.1] * 1536)
        
        service = EmailStyleService()
        service.store = mock_qdrant
        service.embedder = mock_embed
        service._initialized = True  # Skip ensure_collection
        
        result = await service.get_similar_emails("project update")
        
        assert len(result) == 1
        assert result[0]["subject"] == "Q1 Update"
        assert result[0]["score"] == 0.92


class TestGetStylePatterns:
    """Test style pattern aggregation."""
    
    @pytest.mark.asyncio
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    async def test_get_style_patterns_empty_returns_defaults(self, mock_embed_cls, mock_qdrant_cls):
        """Test aggregation returns defaults when no emails exist."""
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.client = MagicMock()
        mock_qdrant.client.collection_exists = MagicMock(return_value=True)
        mock_qdrant.client.scroll = MagicMock(return_value=([], None))
        mock_qdrant.ensure_collection = MagicMock()
        
        mock_embed = MagicMock()
        mock_embed_cls.return_value = mock_embed
        mock_embed.embed_document = AsyncMock(return_value=[0.1] * 1536)
        
        service = EmailStyleService()
        service.store = mock_qdrant
        service.embedder = mock_embed
        service._initialized = True  # Skip ensure_collection
        
        # Call the method - it should return defaults on empty
        patterns = await service.get_style_patterns()
        
        # Should return patterns dict (either real or defaults)
        assert isinstance(patterns, dict)
        assert "common_greetings" in patterns or "sample_count" in patterns


class TestBatchIndexing:
    """Test batch email indexing."""
    
    @pytest.mark.asyncio
    @patch("app.services.email_style_service.QdrantStore")
    @patch("app.services.email_style_service.OllamaEmbeddingClient")
    async def test_batch_index_success(self, mock_embed_cls, mock_qdrant_cls):
        """Test batch indexing multiple emails."""
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.client = MagicMock()
        mock_qdrant.client.collection_exists = MagicMock(return_value=True)
        mock_qdrant.upsert = MagicMock()
        mock_qdrant.ensure_collection = MagicMock()
        
        mock_embed = MagicMock()
        mock_embed_cls.return_value = mock_embed
        mock_embed.embed_document = AsyncMock(return_value=[0.1] * 1536)
        
        service = EmailStyleService()
        service.store = mock_qdrant
        service.embedder = mock_embed
        service._initialized = True  # Skip ensure_collection
        
        emails = [
            {"content": "Hi,\nTest 1\nBest,\nBunmi", "subject": "Email 1"},
            {"content": "Hi,\nTest 2\nBest,\nBunmi", "subject": "Email 2"},
        ]
        
        result = await service.index_emails_batch(emails)
        
        assert result["total"] == 2
        assert "success" in result
        assert "failed" in result
