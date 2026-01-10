"""
Authentication schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional


class GoogleTokenRequest(BaseModel):
    """Request body for Google token exchange."""
    token: str = Field(..., description="Google ID token from frontend")


class RefreshTokenRequest(BaseModel):
    """Request body for token refresh."""
    refresh_token: str = Field(..., description="Refresh token to exchange for new access token")


class TokenResponse(BaseModel):
    """Response containing JWT tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until access token expires


class AuthResponse(BaseModel):
    """Response after successful authentication."""
    success: bool = True
    message: str = "Authentication successful"
    user: "UserResponse"
    tokens: TokenResponse


class UserResponse(BaseModel):
    """User information in auth response."""
    id: str
    email: str
    name: Optional[str] = None
    picture_url: Optional[str] = None
    created_at: Optional[str] = None
    last_login_at: Optional[str] = None
    
    model_config = {"from_attributes": True}


# Update forward reference
AuthResponse.model_rebuild()
