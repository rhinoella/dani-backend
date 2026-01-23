"""
Infographic model for storing generated infographic metadata.

Links to S3-stored images and preserves the structured data,
sources, and generation parameters for each infographic.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Text, Integer, Float, DateTime, Boolean, Index, func, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

from app.database.models.base import Base

if TYPE_CHECKING:
    from app.database.models.user import User


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class InfographicStyle(str, Enum):
    """Infographic visual styles."""
    MODERN = "modern"
    CORPORATE = "corporate"
    MINIMAL = "minimal"
    VIBRANT = "vibrant"
    DARK = "dark"


class InfographicStatus(str, Enum):
    """Infographic generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Infographic(Base):
    """
    Infographic model for tracking generated visuals.
    
    Stores metadata about the infographic including:
    - The original request/topic
    - Extracted structured data (headline, stats, key points)
    - Image storage location in S3
    - Sources used from RAG retrieval
    - Generation timing and confidence
    """
    
    __tablename__ = "infographics"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )
    
    # User who generated the infographic (optional for anonymous)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        comment="User who generated the infographic",
    )
    
    # Request details
    request: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original user request for the infographic",
    )
    topic: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Topic used for RAG search",
    )
    
    # Style and dimensions
    style: Mapped[InfographicStyle] = mapped_column(
        SQLEnum(InfographicStyle, name="infographic_style", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=InfographicStyle.MODERN,
        comment="Visual style of the infographic",
    )
    width: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1024,
        comment="Image width in pixels",
    )
    height: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1024,
        comment="Image height in pixels",
    )
    
    # Structured content extracted by LLM
    headline: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
        comment="Main headline of the infographic",
    )
    subtitle: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Subtitle or context line",
    )
    structured_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Full structured data (stats, key_points, etc.)",
    )
    
    # S3 storage
    s3_key: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
        comment="S3 object key for the image",
    )
    s3_bucket: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="S3 bucket name",
    )
    image_url: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
        comment="Direct URL to the image (may be presigned)",
    )
    image_format: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="png",
        comment="Image format (png, jpg, etc.)",
    )
    image_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Image file size in bytes",
    )
    
    # Sources and confidence
    sources: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="RAG sources used (title, date, score)",
    )
    chunks_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of RAG chunks used",
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="RAG confidence score (0-1)",
    )
    confidence_level: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Confidence level: high, medium, low",
    )
    
    # Timing
    retrieval_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time spent on RAG retrieval",
    )
    extraction_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time spent on LLM data extraction",
    )
    image_gen_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time spent on image generation",
    )
    total_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Total generation time",
    )
    
    # Status
    status: Mapped[InfographicStatus] = mapped_column(
        SQLEnum(InfographicStatus, name="infographic_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=InfographicStatus.PENDING,
        index=True,
        comment="Generation status",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if generation failed",
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    # Optional: soft delete
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Soft delete timestamp",
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_infographics_user_created", "user_id", "created_at"),
        Index("ix_infographics_status_created", "status", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Infographic(id={self.id}, headline='{self.headline}', status={self.status})>"
    
    @property
    def is_completed(self) -> bool:
        """Check if infographic generation completed successfully."""
        return self.status == InfographicStatus.COMPLETED
    
    @property
    def has_image(self) -> bool:
        """Check if infographic has an associated image."""
        return self.s3_key is not None
