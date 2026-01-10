"""
Tests for API dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.api.deps import (
    get_db,
    get_rate_limiter,
    get_conversation_cache,
    get_user_conversations_cache,
    get_google_user,
    get_current_user,
    get_optional_user,
    check_rate_limit,
    increment_rate_limit,
    RateLimitHeaders,
)
from app.core.auth import GoogleUser
from app.cache.rate_limiter import RateLimitResult


class TestGetDb:
    """Tests for get_db dependency."""
    
    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        """Test that get_db yields a session."""
        mock_session = AsyncMock()
        
        with patch("app.api.deps.get_async_session") as mock_get_session:
            async def mock_generator():
                yield mock_session
            
            mock_get_session.return_value = mock_generator()
            
            async for session in get_db():
                assert session == mock_session
                break


class TestGetRateLimiter:
    """Tests for get_rate_limiter dependency."""
    
    @pytest.mark.asyncio
    async def test_get_rate_limiter_returns_limiter(self):
        """Test that get_rate_limiter returns the global limiter."""
        limiter = await get_rate_limiter()
        
        from app.cache.rate_limiter import rate_limiter
        assert limiter is rate_limiter


class TestGetConversationCache:
    """Tests for get_conversation_cache dependency."""
    
    @pytest.mark.asyncio
    async def test_get_conversation_cache_success(self):
        """Test getting conversation cache when Redis available."""
        mock_client = AsyncMock()
        
        with patch("app.api.deps.get_redis_client", return_value=mock_client):
            result = await get_conversation_cache()
            # Should return the cache instance
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_conversation_cache_redis_unavailable(self):
        """Test getting conversation cache when Redis unavailable."""
        with patch("app.api.deps.get_redis_client", side_effect=Exception("Connection refused")):
            result = await get_conversation_cache()
            
            assert result is None


class TestGetUserConversationsCache:
    """Tests for get_user_conversations_cache dependency."""
    
    @pytest.mark.asyncio
    async def test_get_user_conversations_cache_success(self):
        """Test getting user conversations cache when Redis available."""
        mock_client = AsyncMock()
        
        with patch("app.api.deps.get_redis_client", return_value=mock_client):
            result = await get_user_conversations_cache()
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_user_conversations_cache_redis_unavailable(self):
        """Test getting cache when Redis unavailable."""
        with patch("app.api.deps.get_redis_client", side_effect=Exception("Timeout")):
            result = await get_user_conversations_cache()
            
            assert result is None


class TestGetGoogleUser:
    """Tests for get_google_user dependency."""
    
    @pytest.mark.asyncio
    async def test_get_google_user_no_credentials(self):
        """Test that missing credentials raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await get_google_user(credentials=None)
        
        assert exc_info.value.status_code == 401
        assert "Authentication required" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_get_google_user_dev_mode_no_config(self):
        """Test dev mode returns mock user when not configured."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "test-token"
        
        with patch("app.api.deps.settings") as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = "__MISSING__"
            mock_settings.ENV = "development"
            
            result = await get_google_user(credentials=mock_credentials)
            
            assert result.google_id == "dev-user-123"
            assert result.email == "dev@example.com"
    
    @pytest.mark.asyncio
    async def test_get_google_user_prod_mode_no_config(self):
        """Test production mode raises 500 when not configured."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "test-token"
        
        with patch("app.api.deps.settings") as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = "__MISSING__"
            mock_settings.ENV = "production"
            
            with pytest.raises(HTTPException) as exc_info:
                await get_google_user(credentials=mock_credentials)
            
            assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_get_google_user_valid_token(self):
        """Test valid token returns GoogleUser."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "valid-token"
        
        mock_google_user = GoogleUser(
            google_id="123",
            email="test@example.com",
            name="Test User",
            picture_url=None,
            email_verified=True,
        )
        
        with patch("app.api.deps.settings") as mock_settings, \
             patch("app.api.deps.GoogleAuthVerifier") as MockVerifier:
            mock_settings.GOOGLE_CLIENT_ID = "client-id"
            mock_verifier = MockVerifier.return_value
            mock_verifier.verify_token.return_value = mock_google_user
            
            result = await get_google_user(credentials=mock_credentials)
            
            assert result == mock_google_user
    
    @pytest.mark.asyncio
    async def test_get_google_user_invalid_token(self):
        """Test invalid token raises 401."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "invalid-token"
        
        from app.core.auth import GoogleAuthError
        
        with patch("app.api.deps.settings") as mock_settings, \
             patch("app.api.deps.GoogleAuthVerifier") as MockVerifier:
            mock_settings.GOOGLE_CLIENT_ID = "client-id"
            mock_verifier = MockVerifier.return_value
            mock_verifier.verify_token.side_effect = GoogleAuthError("Invalid token")
            
            with pytest.raises(HTTPException) as exc_info:
                await get_google_user(credentials=mock_credentials)
            
            assert exc_info.value.status_code == 401


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(self):
        """Test getting current user creates/updates user."""
        mock_google_user = GoogleUser(
            google_id="123",
            email="test@example.com",
            name="Test User",
            picture_url=None,
            email_verified=True,
        )
        
        mock_db = AsyncMock()
        mock_user = MagicMock()
        mock_user.id = "user-123"
        
        with patch("app.api.deps.UserRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.create_or_update_from_google = AsyncMock(return_value=mock_user)
            
            result = await get_current_user(
                google_user=mock_google_user,
                db=mock_db
            )
            
            assert result == mock_user
            mock_repo.create_or_update_from_google.assert_called_once_with(
                google_id="123",
                email="test@example.com",
                name="Test User",
                picture_url=None,
            )
            mock_db.commit.assert_called_once()


class TestGetOptionalUser:
    """Tests for get_optional_user dependency."""
    
    @pytest.mark.asyncio
    async def test_get_optional_user_no_credentials(self):
        """Test no credentials returns None."""
        mock_db = AsyncMock()
        
        result = await get_optional_user(credentials=None, db=mock_db)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_optional_user_dev_mode(self):
        """Test dev mode returns dev user."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "test-token"
        mock_db = AsyncMock()
        mock_user = MagicMock()
        
        with patch("app.api.deps.settings") as mock_settings, \
             patch("app.api.deps.UserRepository") as MockRepo:
            mock_settings.GOOGLE_CLIENT_ID = "__MISSING__"
            mock_settings.ENV = "development"
            
            mock_repo = MockRepo.return_value
            mock_repo.get_by_google_id = AsyncMock(return_value=mock_user)
            
            result = await get_optional_user(credentials=mock_credentials, db=mock_db)
            
            assert result == mock_user
    
    @pytest.mark.asyncio
    async def test_get_optional_user_dev_mode_creates_user(self):
        """Test dev mode creates user if not exists."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "test-token"
        mock_db = AsyncMock()
        mock_user = MagicMock()
        
        with patch("app.api.deps.settings") as mock_settings, \
             patch("app.api.deps.UserRepository") as MockRepo:
            mock_settings.GOOGLE_CLIENT_ID = "__MISSING__"
            mock_settings.ENV = "development"
            
            mock_repo = MockRepo.return_value
            mock_repo.get_by_google_id = AsyncMock(return_value=None)
            mock_repo.create_or_update_from_google = AsyncMock(return_value=mock_user)
            
            result = await get_optional_user(credentials=mock_credentials, db=mock_db)
            
            assert result == mock_user
            mock_repo.create_or_update_from_google.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_optional_user_invalid_token(self):
        """Test invalid token returns None instead of raising."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "invalid-token"
        mock_db = AsyncMock()
        
        with patch("app.api.deps.settings") as mock_settings, \
             patch("app.api.deps.GoogleAuthVerifier") as MockVerifier:
            mock_settings.GOOGLE_CLIENT_ID = "client-id"
            mock_verifier = MockVerifier.return_value
            mock_verifier.verify_token_lenient.side_effect = Exception("Invalid")
            
            result = await get_optional_user(credentials=mock_credentials, db=mock_db)
            
            assert result is None


class TestCheckRateLimit:
    """Tests for check_rate_limit dependency."""
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self):
        """Test rate limit check passes when allowed."""
        mock_user = MagicMock()
        mock_user.id = "user-123"
        
        mock_result = RateLimitResult(
            allowed=True,
            remaining=10,
            reset_at=1234567890.0,
            limit=20
        )
        
        with patch("app.api.deps.rate_limiter") as mock_limiter:
            mock_limiter.check_rate_limit = AsyncMock(return_value=mock_result)
            mock_limiter.increment = AsyncMock()
            
            result = await check_rate_limit(user=mock_user)
            
            assert result == mock_user
            mock_limiter.increment.assert_called_once_with("user-123")
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self):
        """Test rate limit check raises 429 when exceeded."""
        mock_user = MagicMock()
        mock_user.id = "user-123"
        
        mock_result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_at=1234567890.0,
            limit=20,
            retry_after=30
        )
        
        with patch("app.api.deps.rate_limiter") as mock_limiter:
            mock_limiter.check_rate_limit = AsyncMock(return_value=mock_result)
            
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(user=mock_user)
            
            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in str(exc_info.value.detail)
            assert exc_info.value.headers["Retry-After"] == "30"


class TestIncrementRateLimit:
    """Tests for increment_rate_limit function."""
    
    @pytest.mark.asyncio
    async def test_increment_rate_limit(self):
        """Test incrementing rate limit."""
        with patch("app.api.deps.rate_limiter") as mock_limiter:
            mock_limiter.increment = AsyncMock()
            
            await increment_rate_limit("user-123")
            
            mock_limiter.increment.assert_called_once_with("user-123")


class TestRateLimitHeaders:
    """Tests for RateLimitHeaders helper."""
    
    @pytest.mark.asyncio
    async def test_get_headers(self):
        """Test getting rate limit headers."""
        mock_result = RateLimitResult(
            allowed=True,
            remaining=15,
            reset_at=1234567890.0,
            limit=20
        )
        
        with patch("app.api.deps.rate_limiter") as mock_limiter:
            mock_limiter.check_rate_limit = AsyncMock(return_value=mock_result)
            
            headers = await RateLimitHeaders.get_headers("user-123")
            
            assert headers["X-RateLimit-Limit"] == "20"
            assert headers["X-RateLimit-Remaining"] == "15"
            assert headers["X-RateLimit-Reset"] == "1234567890"
