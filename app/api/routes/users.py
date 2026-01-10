"""
User routes.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.api.deps import get_db, get_current_user, get_rate_limiter
from app.schemas.user import UserResponse, UserUpdate, UserProfileResponse, UserRateLimitResponse
from app.services.user_service import UserService
from app.database.models import User
from app.cache.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """Get the current user's profile with statistics."""
    user_service = UserService(db, rate_limiter)
    profile = await user_service.get_user_profile(str(current_user.id))
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserProfileResponse(**profile)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    update: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update the current user's profile."""
    user_service = UserService(db)
    
    updated_user = await user_service.update_user(
        str(current_user.id),
        name=update.name,
        picture_url=update.picture_url
    )
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=str(updated_user.id),
        email=updated_user.email,
        name=updated_user.name,
        picture_url=updated_user.picture_url,
        created_at=updated_user.created_at,
        last_login_at=updated_user.last_login_at
    )


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete the current user's account.
    This is a soft delete - data can be recovered.
    """
    user_service = UserService(db)
    success = await user_service.delete_user(str(current_user.id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
    
    logger.info(f"User account deleted: {current_user.email}")


@router.get("/me/rate-limit", response_model=UserRateLimitResponse)
async def get_rate_limit_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """Get the current user's rate limit status."""
    user_service = UserService(db, rate_limiter)
    rate_status = await user_service.get_rate_limit_status(str(current_user.id))
    
    return UserRateLimitResponse(**rate_status)
