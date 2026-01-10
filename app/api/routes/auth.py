"""
Authentication routes.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.api.deps import get_db, get_rate_limiter, get_current_user
from app.schemas.auth import (
    GoogleTokenRequest, 
    AuthResponse, 
    UserResponse, 
    TokenResponse,
    RefreshTokenRequest
)
from app.core.auth import GoogleAuthVerifier, GoogleAuthError
from app.core.tokens import (
    create_token_pair, 
    verify_refresh_token, 
    TokenError,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.services.user_service import UserService
from app.cache.rate_limiter import RateLimiter
from app.database.models import User
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/google", response_model=AuthResponse)
async def authenticate_google(
    request: GoogleTokenRequest,
    db: AsyncSession = Depends(get_db),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """
    Authenticate user with Google ID token.
    
    Exchanges Google ID token for application JWT tokens.
    Returns:
    - access_token: Short-lived token for API calls (30 min)
    - refresh_token: Long-lived token to get new access tokens (7 days)
    """
    try:
        # Verify Google token
        verifier = GoogleAuthVerifier()
        google_user = verifier.verify_token(request.token)
        
        # Get or create user
        user_service = UserService(db, rate_limiter)
        user = await user_service.get_or_create_from_google(google_user)
        
        # Create JWT token pair
        tokens = create_token_pair(str(user.id), user.email)
        
        logger.info(f"User authenticated: {user.email}")
        
        return AuthResponse(
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                name=user.name,
                picture_url=user.picture_url,
                created_at=user.created_at.isoformat() if user.created_at else None,
                last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
            ),
            tokens=TokenResponse(**tokens)
        )
        
    except GoogleAuthError as e:
        logger.warning(f"Google auth failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Exchange a refresh token for a new access token.
    
    Use this when the access token expires (after ~30 minutes).
    The refresh token itself is valid for 7 days.
    """
    try:
        # Verify refresh token
        payload = verify_refresh_token(request.refresh_token)
        
        # Verify user still exists
        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(payload.sub)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Issue new token pair
        tokens = create_token_pair(str(user.id), user.email)
        
        logger.info(f"Token refreshed for user: {user.email}")
        
        return TokenResponse(**tokens)
        
    except TokenError as e:
        logger.warning(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/verify")
async def verify_token_endpoint(
    request: GoogleTokenRequest
):
    """
    Verify a Google token without creating a user.
    Useful for checking if a token is still valid.
    """
    try:
        verifier = GoogleAuthVerifier()
        google_user = verifier.verify_token(request.token)
        
        return {
            "valid": True,
            "email": google_user.email,
            "expires_in": None  # Google tokens don't have a simple expiry
        }
        
    except GoogleAuthError as e:
        return {
            "valid": False,
            "error": str(e)
        }


@router.get("/me")
async def get_current_user_info(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get current authenticated user's information."""
    user_service = UserService(db)
    profile = await user_service.get_user_profile(str(user.id))
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return profile
