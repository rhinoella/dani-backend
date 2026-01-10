"""
User schemas.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """Base user fields."""
    email: EmailStr
    name: Optional[str] = None
    picture_url: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a user."""
    google_id: str


class UserCreateManual(BaseModel):
    """Schema for manually creating a user."""
    email: EmailStr
    name: str


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    name: Optional[str] = None
    picture_url: Optional[str] = None


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    name: Optional[str] = None
    picture_url: Optional[str] = None
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


class UserProfileResponse(UserResponse):
    """Extended user profile response."""
    conversation_count: int = 0
    message_count: int = 0


class UserRateLimitResponse(BaseModel):
    """User rate limit information."""
    minute: "RateLimitInfo"
    day: "RateLimitInfo"


class RateLimitInfo(BaseModel):
    """Rate limit details."""
    used: int
    limit: int
    reset_in: int = Field(..., description="Seconds until reset")


# Update forward reference
UserRateLimitResponse.model_rebuild()
