"""
JWT Token management for DANI Engine.

Issues and verifies application-specific JWTs for authentication.
This replaces direct use of Google ID tokens for API calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from dataclasses import dataclass
import secrets

import jwt

from app.core.config import settings

logger = logging.getLogger(__name__)

# Token configuration
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Short-lived
REFRESH_TOKEN_EXPIRE_DAYS = 7    # Long-lived
ALGORITHM = "HS256"


@dataclass
class TokenPayload:
    """Decoded token payload."""
    sub: str  # User ID
    email: str
    type: str  # "access" or "refresh"
    exp: datetime
    iat: datetime
    jti: str  # Unique token ID


class TokenError(Exception):
    """Exception raised for token errors."""
    pass


def get_jwt_secret() -> str:
    """Get JWT secret key from settings or generate one."""
    secret = getattr(settings, 'JWT_SECRET_KEY', None)
    if not secret or secret == "__MISSING__":
        # In development, use a deterministic secret (NOT for production!)
        if settings.ENV == "development":
            logger.warning("Using development JWT secret - NOT SECURE FOR PRODUCTION")
            return "dev-secret-key-change-in-production-" + settings.APP_NAME
        raise TokenError("JWT_SECRET_KEY must be set in production")
    return secret


def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a short-lived access token.
    
    Args:
        user_id: User's UUID as string
        email: User's email
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT access token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16),  # Unique token ID
    }
    
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def create_refresh_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a long-lived refresh token.
    
    Args:
        user_id: User's UUID as string
        email: User's email
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": user_id,
        "email": email,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16),
    }
    
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def create_token_pair(user_id: str, email: str) -> dict:
    """
    Create both access and refresh tokens.
    
    Args:
        user_id: User's UUID as string
        email: User's email
        
    Returns:
        Dict with access_token, refresh_token, token_type, and expires_in
    """
    access_token = create_access_token(user_id, email)
    refresh_token = create_refresh_token(user_id, email)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # In seconds
    }


def verify_token(token: str, expected_type: str = "access") -> TokenPayload:
    """
    Verify and decode a JWT token.
    
    Args:
        token: The JWT token to verify
        expected_type: Expected token type ("access" or "refresh")
        
    Returns:
        TokenPayload with decoded information
        
    Raises:
        TokenError: If token is invalid, expired, or wrong type
    """
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[ALGORITHM],
        )
        
        # Verify token type
        if payload.get("type") != expected_type:
            raise TokenError(f"Invalid token type. Expected {expected_type}")
        
        return TokenPayload(
            sub=payload["sub"],
            email=payload["email"],
            type=payload["type"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            jti=payload["jti"],
        )
        
    except jwt.ExpiredSignatureError:
        raise TokenError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise TokenError(f"Invalid token: {str(e)}")
    except KeyError as e:
        raise TokenError(f"Token missing required field: {str(e)}")


def verify_access_token(token: str) -> TokenPayload:
    """Verify an access token."""
    return verify_token(token, "access")


def verify_refresh_token(token: str) -> TokenPayload:
    """Verify a refresh token."""
    return verify_token(token, "refresh")
