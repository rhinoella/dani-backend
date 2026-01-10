"""
RAG Log model for tracking RAG pipeline interactions.

Stores analytics data for:
- Query analysis and debugging
- Performance monitoring
- Quality evaluation
- User feedback collection
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Text, Float, Integer, DateTime, Boolean, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

from app.database.models.base import Base

if TYPE_CHECKING:
    from app.database.models.user import User
    from app.database.models.conversation import Conversation


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class RAGLog(Base):
    """
    RAG interaction log for analytics and debugging.
    
    Tracks every RAG pipeline execution with detailed metrics.
    """
    
    __tablename__ = "rag_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Optional associations
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Query information
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    query_length: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    query_intent: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Detected intent: factual, comparative, temporal, etc.",
    )
    query_entities: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Extracted entities from query",
    )
    
    # Retrieval metrics
    chunks_retrieved: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Number of chunks retrieved",
    )
    chunks_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Number of chunks actually used in prompt",
    )
    retrieval_scores: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Similarity scores for retrieved chunks",
    )
    sources: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Source documents/meetings used",
    )
    
    # Response information
    answer_length: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    output_format: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Requested output format: summary, decisions, etc.",
    )
    
    # Confidence metrics
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Overall confidence score (0-1)",
    )
    confidence_level: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="high, medium, low, very_low, none",
    )
    confidence_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Performance timing (all in milliseconds)
    embedding_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    retrieval_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    generation_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    total_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    
    # Cache information
    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    cache_type: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="semantic, exact, embedding, none",
    )
    
    # Error tracking
    success: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )
    error_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # User feedback (can be updated later)
    user_rating: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="1-5 star rating or thumbs up/down (1/0)",
    )
    user_feedback: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Optional text feedback",
    )
    feedback_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Metadata
    model_used: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="LLM model used for generation",
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
        comment="Additional metadata, filters used, etc.",
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="rag_logs",
        lazy="selectin",
    )
    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation",
        back_populates="rag_logs",
        lazy="selectin",
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_rag_logs_created_at", "created_at"),
        Index("ix_rag_logs_success", "success"),
        Index("ix_rag_logs_confidence_level", "confidence_level"),
        Index("ix_rag_logs_cache_hit", "cache_hit"),
        Index("ix_rag_logs_query_intent", "query_intent"),
    )
    
    def __repr__(self) -> str:
        return (
            f"<RAGLog(id={self.id}, "
            f"query={self.query[:30]}..., "
            f"success={self.success}, "
            f"confidence={self.confidence_score})>"
        )
