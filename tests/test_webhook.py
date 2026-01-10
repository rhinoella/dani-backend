"""
Tests for webhook functionality.

Tests cover:
- Webhook payload validation
- Signature verification
- Event handling
- Integration with ingestion service
"""

import pytest
import hmac
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.services.webhook_service import WebhookService
from app.schemas.webhook import (
    FirefliesWebhookPayload,
    WebhookEventType,
    WebhookResponse,
)


class TestWebhookPayloadValidation:
    """Tests for webhook payload validation."""
    
    def test_valid_payload(self):
        """Should accept valid payload."""
        payload = FirefliesWebhookPayload(
            meetingId="abc123",
            eventType="Transcription completed",
            title="Team Standup",
            organizerEmail="user@example.com",
        )
        
        assert payload.meetingId == "abc123"
        assert payload.eventType == "Transcription completed"
        assert payload.title == "Team Standup"
    
    def test_minimal_payload(self):
        """Should accept minimal required fields."""
        payload = FirefliesWebhookPayload(
            meetingId="abc123",
            eventType="Transcription completed",
        )
        
        assert payload.meetingId == "abc123"
        assert payload.title is None
        assert payload.organizerEmail is None
    
    def test_payload_with_extra_fields(self):
        """Should accept extra fields (Fireflies may add new fields)."""
        data = {
            "meetingId": "abc123",
            "eventType": "Transcription completed",
            "newField": "some value",
            "anotherField": 123,
        }
        
        payload = FirefliesWebhookPayload(**data)
        assert payload.meetingId == "abc123"
    
    def test_missing_required_fields(self):
        """Should reject payload missing required fields."""
        with pytest.raises(Exception):  # Pydantic validation error
            FirefliesWebhookPayload(eventType="Transcription completed")
        
        with pytest.raises(Exception):
            FirefliesWebhookPayload(meetingId="abc123")


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebhookService()
    
    def test_signature_verification_disabled(self):
        """Should pass when no secret configured (dev mode)."""
        with patch.object(self.service, 'ingestion_service'):
            # Simulate no secret configured
            with patch('app.services.webhook_service.settings') as mock_settings:
                mock_settings.FIREFLIES_WEBHOOK_SECRET = "__MISSING__"
                
                result = self.service.verify_signature(b"any payload", None)
                assert result is True
    
    def test_signature_verification_no_signature_provided(self):
        """Should fail when secret configured but no signature in request."""
        with patch('app.services.webhook_service.settings') as mock_settings:
            mock_settings.FIREFLIES_WEBHOOK_SECRET = "my-secret-key"
            
            result = self.service.verify_signature(b"payload", None)
            assert result is False
    
    def test_signature_verification_valid(self):
        """Should pass with valid signature."""
        secret = "my-secret-key"
        payload = b'{"meetingId": "abc123", "eventType": "Transcription completed"}'
        
        # Generate valid signature
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        with patch('app.services.webhook_service.settings') as mock_settings:
            mock_settings.FIREFLIES_WEBHOOK_SECRET = secret
            
            result = self.service.verify_signature(payload, expected_signature)
            assert result is True
    
    def test_signature_verification_invalid(self):
        """Should fail with invalid signature."""
        with patch('app.services.webhook_service.settings') as mock_settings:
            mock_settings.FIREFLIES_WEBHOOK_SECRET = "my-secret-key"
            
            payload = b'{"meetingId": "abc123"}'
            invalid_signature = "invalid_signature_here"
            
            result = self.service.verify_signature(payload, invalid_signature)
            assert result is False


class TestWebhookEventHandling:
    """Tests for webhook event handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = WebhookService()
    
    @pytest.mark.asyncio
    async def test_transcription_completed_event(self):
        """Should process transcription completed event."""
        # Mock the ingestion service
        self.service.ingestion_service = MagicMock()
        self.service.ingestion_service.ingest_transcript = AsyncMock(
            return_value={"ingested": 10, "transcript_id": "abc123"}
        )
        
        payload = FirefliesWebhookPayload(
            meetingId="abc123",
            eventType="Transcription completed",
            title="Team Meeting",
            organizerEmail="test@example.com",
        )
        
        result = await self.service.process_webhook(payload)
        
        assert result["success"] is True
        assert result["transcript_id"] == "abc123"
        assert result["chunks_created"] == 10
        
        # Verify ingestion was called
        self.service.ingestion_service.ingest_transcript.assert_called_once_with("abc123")
    
    @pytest.mark.asyncio
    async def test_transcription_completed_ingestion_failure(self):
        """Should handle ingestion failure gracefully."""
        self.service.ingestion_service = MagicMock()
        self.service.ingestion_service.ingest_transcript = AsyncMock(
            side_effect=Exception("Embedding service unavailable")
        )
        
        payload = FirefliesWebhookPayload(
            meetingId="abc123",
            eventType="Transcription completed",
        )
        
        result = await self.service.process_webhook(payload)
        
        assert result["success"] is False
        assert "Failed to ingest" in result["message"]
        assert result["chunks_created"] == 0
    
    @pytest.mark.asyncio
    async def test_unknown_event_type(self):
        """Should acknowledge unknown event types."""
        payload = FirefliesWebhookPayload(
            meetingId="abc123",
            eventType="Meeting scheduled",  # Unknown event
        )
        
        result = await self.service.process_webhook(payload)
        
        assert result["success"] is True
        assert "acknowledged but not processed" in result["message"]
    
    def test_webhook_enabled_check(self):
        """Should correctly check if webhook is enabled."""
        with patch('app.services.webhook_service.settings') as mock_settings:
            # Enabled - has valid API key
            mock_settings.FIREFLIES_API_KEY = "valid-api-key"
            assert self.service.is_webhook_enabled() is True
            
            # Disabled - missing API key
            mock_settings.FIREFLIES_API_KEY = "__MISSING__"
            assert self.service.is_webhook_enabled() is False
            
            # Disabled - empty API key
            mock_settings.FIREFLIES_API_KEY = ""
            assert self.service.is_webhook_enabled() is False


class TestWebhookAPIEndpoints:
    """Tests for webhook API endpoints."""
    
    @pytest.fixture
    def mock_webhook_service(self):
        """Create a mock webhook service."""
        with patch('app.api.routes.webhook.webhook_service') as mock:
            mock.is_webhook_enabled.return_value = True
            mock.verify_signature.return_value = True
            mock.process_webhook = AsyncMock(return_value={
                "success": True,
                "message": "Transcript ingested",
                "transcript_id": "abc123",
                "chunks_created": 10,
            })
            yield mock
    
    def test_webhook_health_enabled(self):
        """Should return healthy status when enabled."""
        from app.main import app
        
        with patch('app.api.routes.webhook.webhook_service') as mock:
            mock.is_webhook_enabled.return_value = True
            
            client = TestClient(app)
            response = client.get("/api/v1/webhook/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["webhook_enabled"] is True
    
    def test_webhook_health_disabled(self):
        """Should indicate disabled status when API key missing."""
        from app.main import app
        
        with patch('app.api.routes.webhook.webhook_service') as mock:
            mock.is_webhook_enabled.return_value = False
            
            client = TestClient(app)
            response = client.get("/api/v1/webhook/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["webhook_enabled"] is False
    
    def test_fireflies_webhook_success(self, mock_webhook_service):
        """Should process valid webhook successfully."""
        from app.main import app
        
        client = TestClient(app)
        
        payload = {
            "meetingId": "abc123",
            "eventType": "Transcription completed",
            "title": "Team Meeting",
        }
        
        response = client.post(
            "/api/v1/webhook/fireflies",
            json=payload,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["transcript_id"] == "abc123"
    
    def test_fireflies_webhook_disabled(self):
        """Should return 503 when webhook is disabled."""
        from app.main import app
        
        with patch('app.api.routes.webhook.webhook_service') as mock:
            mock.is_webhook_enabled.return_value = False
            
            client = TestClient(app)
            
            payload = {
                "meetingId": "abc123",
                "eventType": "Transcription completed",
            }
            
            response = client.post(
                "/api/v1/webhook/fireflies",
                json=payload,
            )
            
            assert response.status_code == 503
    
    def test_fireflies_webhook_invalid_signature(self):
        """Should return 401 when signature verification fails."""
        from app.main import app
        
        with patch('app.api.routes.webhook.webhook_service') as mock:
            mock.is_webhook_enabled.return_value = True
            mock.verify_signature.return_value = False
            
            client = TestClient(app)
            
            payload = {
                "meetingId": "abc123",
                "eventType": "Transcription completed",
            }
            
            response = client.post(
                "/api/v1/webhook/fireflies",
                json=payload,
                headers={"X-Fireflies-Signature": "invalid"},
            )
            
            assert response.status_code == 401
    
    def test_fireflies_webhook_invalid_payload(self, mock_webhook_service):
        """Should return 400 for invalid payload."""
        from app.main import app
        
        client = TestClient(app)
        
        # Missing required fields
        payload = {"title": "Some meeting"}
        
        response = client.post(
            "/api/v1/webhook/fireflies",
            json=payload,
        )
        
        assert response.status_code == 400 or response.status_code == 422
    
    def test_fireflies_test_webhook(self, mock_webhook_service):
        """Should process test webhook endpoint."""
        from app.main import app
        
        client = TestClient(app)
        
        payload = {
            "meetingId": "test123",
            "eventType": "Transcription completed",
            "title": "Test Meeting",
        }
        
        response = client.post(
            "/api/v1/webhook/fireflies/test",
            json=payload,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestWebhookIntegration:
    """Integration tests for webhook with ingestion."""
    
    @pytest.mark.asyncio
    async def test_full_webhook_flow(self):
        """Test complete webhook flow from receipt to ingestion."""
        service = WebhookService()
        
        # Mock the ingestion service's ingest_transcript method
        service.ingestion_service.ingest_transcript = AsyncMock(
            return_value={"ingested": 5, "transcript_id": "test123"}
        )
        
        payload = FirefliesWebhookPayload(
            meetingId="test123",
            eventType="Transcription completed",
        )
        
        result = await service.process_webhook(payload)
        
        assert result["success"] is True
        assert result["transcript_id"] == "test123"
        assert result["chunks_created"] == 5
        
        # Verify ingestion was called with correct transcript ID
        service.ingestion_service.ingest_transcript.assert_called_once_with("test123")
    
    @pytest.mark.asyncio
    async def test_webhook_with_full_payload(self):
        """Test webhook with all optional fields populated."""
        service = WebhookService()
        
        service.ingestion_service.ingest_transcript = AsyncMock(
            return_value={"ingested": 10}
        )
        
        payload = FirefliesWebhookPayload(
            meetingId="full123",
            eventType="Transcription completed",
            title="Full Integration Meeting",
            organizerEmail="organizer@company.com",
            participants=["alice@company.com", "bob@company.com"],
            duration=3600,
            dateTime=1704067200000,
            transcriptUrl="https://app.fireflies.ai/view/full123",
            clientReferenceId="custom-ref-001",
        )
        
        result = await service.process_webhook(payload)
        
        assert result["success"] is True
        assert result["chunks_created"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
