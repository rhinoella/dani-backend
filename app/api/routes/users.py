"""
User routes.
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.api.deps import get_db, get_current_user, get_rate_limiter
from app.schemas.user import UserResponse, UserUpdate, UserProfileResponse, UserRateLimitResponse, UserCreateManual
from app.services.user_service import UserService
from app.database.models import User
from app.cache.rate_limiter import RateLimiter
from typing import List

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


@router.post("/me/avatar", response_model=UserResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a new profile avatar image."""
    import base64
    
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Validate file size (max 1MB for avatars stored in DB)
    max_size = 1 * 1024 * 1024  # 1MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 1MB for avatars."
        )
    
    try:
        # Convert to base64 data URL (stored directly in database)
        base64_data = base64.b64encode(content).decode('utf-8')
        picture_url = f"data:{file.content_type};base64,{base64_data}"
        
        # Update user's picture_url
        user_service = UserService(db)
        updated_user = await user_service.update_user(
            str(current_user.id),
            picture_url=picture_url
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user profile"
            )
        
        logger.info(f"Avatar uploaded for user {current_user.email}")
        
        return UserResponse(
            id=str(updated_user.id),
            email=updated_user.email,
            name=updated_user.name,
            picture_url=updated_user.picture_url,
            created_at=updated_user.created_at,
            last_login_at=updated_user.last_login_at
        )
        
    except Exception as e:
        logger.error(f"Avatar upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload avatar"
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


@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all users (admin only ideally, but open for now)."""
    user_service = UserService(db)
    return await user_service.get_users(skip=skip, limit=limit)


@router.post("/", response_model=UserResponse)
async def create_user(
    user_in: UserCreateManual,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new user manually."""
    user_service = UserService(db)
    try:
        user = await user_service.create_manual_user(email=user_in.email, name=user_in.name)
        return UserResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a user."""
    user_service = UserService(db)
    
    # Check if user exists first? update_user handles it.
    updated = await user_service.update_user(
        user_id,
        name=user_in.name,
        picture_url=user_in.picture_url
    )
    
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
        
    return UserResponse.model_validate(updated)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a user."""
    user_service = UserService(db)
    success = await user_service.delete_user(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or already deleted"
        )
