"""
Webhook routes for Fireflies integration.

These endpoints receive webhook events from Fireflies when
transcripts are ready, allowing automatic ingestion.

To configure in Fireflies:
1. Go to Fireflies Settings > Integrations > Webhooks
2. Add a new webhook with URL: https://your-domain/api/v1/webhook/fireflies
3. Select events: "Transcription completed"
4. (Optional) Add a webhook secret for verification
"""

from fastapi import APIRouter, Request, HTTPException, Header, BackgroundTasks
from typing import Optional
import logging

from app.services.webhook_service import WebhookService
from app.schemas.webhook import (
    FirefliesWebhookPayload,
    WebhookResponse,
    WebhookHealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["Webhooks"])

# Initialize service
webhook_service = WebhookService()


@router.get("/health", response_model=WebhookHealthResponse)
async def webhook_health():
    """
    Check webhook endpoint health and configuration status.
    
    Use this endpoint to verify the webhook is accessible
    and properly configured.
    """
    is_enabled = webhook_service.is_webhook_enabled()
    
    return WebhookHealthResponse(
        status="ok",
        webhook_enabled=is_enabled,
        message="Webhook endpoint is ready" if is_enabled else "Webhook disabled - Fireflies API key not configured"
    )


@router.post("/fireflies", response_model=WebhookResponse)
async def fireflies_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_fireflies_signature: Optional[str] = Header(None, alias="X-Fireflies-Signature"),
):
    """
    Receive Fireflies webhook events.
    
    This endpoint handles webhook notifications from Fireflies
    when transcripts are completed. The transcript is automatically
    ingested into the vector store.
    
    **Webhook Configuration in Fireflies:**
    - URL: `https://your-domain/api/v1/webhook/fireflies`
    - Events: Select "Transcription completed"
    - Secret: (optional) Add for signature verification
    
    **Security:**
    - Optionally verifies webhook signature using FIREFLIES_WEBHOOK_SECRET
    - Processes requests asynchronously to respond quickly
    """
    # Check if webhook is enabled
    if not webhook_service.is_webhook_enabled():
        logger.warning("Webhook received but Fireflies API key not configured")
        raise HTTPException(
            status_code=503,
            detail="Webhook processing disabled - Fireflies API key not configured"
        )
    
    # Get raw body for signature verification
    body = await request.body()
    
    # Verify signature if configured
    if not webhook_service.verify_signature(body, x_fireflies_signature):
        logger.warning("Webhook signature verification failed")
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature"
        )
    
    # Parse payload
    try:
        json_body = await request.json()
        payload = FirefliesWebhookPayload(**json_body)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid webhook payload: {str(e)}"
        )
    
    logger.info(f"Received Fireflies webhook: {payload.eventType} for {payload.meetingId}")
    
    # Process webhook (can be done in background for faster response)
    # For now, process synchronously to return accurate result
    result = await webhook_service.process_webhook(payload)
    
    return WebhookResponse(
        success=result["success"],
        message=result["message"],
        transcript_id=result.get("transcript_id"),
        chunks_created=result.get("chunks_created"),
    )


@router.post("/fireflies/test")
async def test_fireflies_webhook(payload: FirefliesWebhookPayload):
    """
    Test endpoint for simulating Fireflies webhooks.
    
    Use this to test the webhook processing without
    actually receiving a webhook from Fireflies.
    
    **Example payload:**
    ```json
    {
        "meetingId": "abc123",
        "eventType": "Transcription completed",
        "title": "Team Standup",
        "organizerEmail": "user@example.com"
    }
    ```
    """
    logger.info(f"Test webhook received: {payload.eventType} for {payload.meetingId}")
    
    result = await webhook_service.process_webhook(payload)
    
    return WebhookResponse(
        success=result["success"],
        message=result["message"],
        transcript_id=result.get("transcript_id"),
        chunks_created=result.get("chunks_created"),
    )
