"""
Schemas for Fireflies webhook payloads.

Fireflies webhook events:
- Transcription completed: sent when a meeting transcription is ready
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum


class WebhookEventType(str, Enum):
    """Fireflies webhook event types."""
    TRANSCRIPTION_COMPLETED = "Transcription completed"


class WebhookMeetingData(BaseModel):
    """Meeting data sent in webhook payload."""
    id: str = Field(..., description="Fireflies transcript ID")
    title: Optional[str] = Field(None, description="Meeting title")
    organizer_email: Optional[str] = Field(None, description="Meeting organizer email")
    participants: Optional[List[str]] = Field(default_factory=list, description="List of participant emails")
    date: Optional[int] = Field(None, description="Meeting date as epoch timestamp (ms)")
    duration: Optional[int] = Field(None, description="Meeting duration in seconds")
    transcript_url: Optional[str] = Field(None, description="URL to view transcript")


class FirefliesWebhookPayload(BaseModel):
    """
    Fireflies webhook payload structure.
    
    Fireflies sends POST requests with JSON body when events occur.
    The exact structure depends on the webhook configuration.
    """
    meetingId: str = Field(..., description="Fireflies transcript/meeting ID")
    eventType: str = Field(..., description="Type of webhook event")
    clientReferenceId: Optional[str] = Field(None, description="Custom reference ID if set")
    title: Optional[str] = Field(None, description="Meeting title")
    organizerEmail: Optional[str] = Field(None, description="Meeting organizer email")
    participants: Optional[List[str]] = Field(default_factory=list, description="Participant list")
    duration: Optional[int] = Field(None, description="Meeting duration")
    dateTime: Optional[int] = Field(None, description="Meeting timestamp")
    transcriptUrl: Optional[str] = Field(None, description="Transcript URL")
    
    model_config = {"extra": "allow"}  # Allow additional fields from webhook


class WebhookResponse(BaseModel):
    """Response schema for webhook endpoint."""
    success: bool
    message: str
    transcript_id: Optional[str] = None
    chunks_created: Optional[int] = None


class WebhookHealthResponse(BaseModel):
    """Response schema for webhook health check."""
    status: str
    webhook_enabled: bool
    message: str
