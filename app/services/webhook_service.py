"""
Webhook service for handling Fireflies webhook events.

This service processes incoming webhook payloads and triggers
transcript ingestion automatically.
"""

from __future__ import annotations

import logging
import hmac
import hashlib
from typing import Any, Dict, Optional

from app.core.config import settings
from app.services.ingestion_service import IngestionService
from app.schemas.webhook import FirefliesWebhookPayload, WebhookEventType

logger = logging.getLogger(__name__)


class WebhookService:
    """Service for processing Fireflies webhooks."""
    
    def __init__(self) -> None:
        logger.info("Initializing WebhookService")
        self.ingestion_service = IngestionService()
    
    def verify_signature(
        self,
        payload: bytes,
        signature: Optional[str],
    ) -> bool:
        """
        Verify webhook signature if webhook secret is configured.
        
        Fireflies may send a signature header for verification.
        If no secret is configured, we skip verification (development mode).
        
        Args:
            payload: Raw request body bytes
            signature: Signature from request header
            
        Returns:
            True if signature is valid or verification is disabled
        """
        webhook_secret = getattr(settings, 'FIREFLIES_WEBHOOK_SECRET', None)
        
        # Skip verification if no secret configured (development mode)
        if not webhook_secret or webhook_secret == "__MISSING__":
            logger.warning("Webhook signature verification disabled - no secret configured")
            return True
        
        if not signature:
            logger.warning("No signature provided in webhook request")
            return False
        
        # Compute expected signature
        expected_signature = hmac.new(
            webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures securely
        is_valid = hmac.compare_digest(signature, expected_signature)
        
        if not is_valid:
            logger.warning("Webhook signature verification failed")
        
        return is_valid
    
    async def process_webhook(
        self,
        payload: FirefliesWebhookPayload,
    ) -> Dict[str, Any]:
        """
        Process incoming Fireflies webhook.
        
        Args:
            payload: Parsed webhook payload
            
        Returns:
            Processing result with status and details
        """
        logger.info(f"Processing webhook event: {payload.eventType} for meeting: {payload.meetingId}")
        
        # Handle different event types
        if payload.eventType == WebhookEventType.TRANSCRIPTION_COMPLETED.value:
            return await self._handle_transcription_completed(payload)
        else:
            logger.warning(f"Unknown webhook event type: {payload.eventType}")
            return {
                "success": True,
                "message": f"Event type '{payload.eventType}' acknowledged but not processed",
                "transcript_id": payload.meetingId,
            }
    
    async def _handle_transcription_completed(
        self,
        payload: FirefliesWebhookPayload,
    ) -> Dict[str, Any]:
        """
        Handle transcription completed event.
        
        This triggers the ingestion of the new transcript.
        
        Args:
            payload: Webhook payload with meeting details
            
        Returns:
            Ingestion result
        """
        transcript_id = payload.meetingId
        
        logger.info(f"Transcription completed for meeting: {transcript_id}")
        logger.info(f"  Title: {payload.title}")
        logger.info(f"  Organizer: {payload.organizerEmail}")
        
        try:
            # Use the ingestion service to process the transcript
            result = await self.ingestion_service.ingest_transcript(transcript_id)
            
            chunks_created = result.get("ingested", 0)
            
            logger.info(f"Successfully ingested transcript {transcript_id}: {chunks_created} chunks")
            
            return {
                "success": True,
                "message": f"Transcript ingested successfully",
                "transcript_id": transcript_id,
                "chunks_created": chunks_created,
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest transcript {transcript_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to ingest transcript: {str(e)}",
                "transcript_id": transcript_id,
                "chunks_created": 0,
            }
    
    def is_webhook_enabled(self) -> bool:
        """Check if webhook processing is enabled."""
        # Webhook is enabled if we have a Fireflies API key
        api_key = settings.FIREFLIES_API_KEY
        return bool(api_key and api_key != "__MISSING__")
