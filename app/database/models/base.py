"""
SQLAlchemy declarative base.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB


class Base(DeclarativeBase):
    """Base class for all models."""
    
    type_annotation_map = {
        dict: JSONB,
    }
