"""
Message model for storing individual chat messages.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, Index, Float, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

from app.database.models.base import Base

if TYPE_CHECKING:
    from app.database.models.conversation import Conversation


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class Message(Base):
    """
    Message model for storing individual chat messages.
    
    Messages belong to a conversation and include role, content, and metadata.
    """
    
    __tablename__ = "messages"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Message content
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="'user', 'assistant', or 'system'",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    
    # RAG metadata
    sources: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="RAG sources used for this response",
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score for this response",
    )
    
    # Additional metadata
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
        comment="Output format, timings, etc.",
    )
    
    # Token usage
    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Estimated tokens used",
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Soft delete
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        Index("ix_messages_conversation_not_deleted", "conversation_id", "deleted_at"),
    )
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, role={self.role}, content={content_preview})>"
    
    @property
    def is_deleted(self) -> bool:
        """Check if message is soft deleted."""
        return self.deleted_at is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
            "confidence_score": self.confidence_score,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
