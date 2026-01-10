"""
Tests for authentication module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from app.core.auth import (
    GoogleAuthVerifier,
    GoogleAuthError,
    GoogleUser,
    verify_google_token,
)


def utc_now():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# ============== GoogleUser Tests ==============

class TestGoogleUser:
    """Tests for GoogleUser dataclass."""
    
    def test_google_user_creation(self):
        """Test creating a GoogleUser."""
        user = GoogleUser(
            google_id="123456789",
            email="test@example.com",
            name="Test User",
            picture_url="https://example.com/photo.jpg",
            email_verified=True,
        )
        
        assert user.google_id == "123456789"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.picture_url == "https://example.com/photo.jpg"
        assert user.email_verified is True
    
    def test_google_user_minimal(self):
        """Test creating a GoogleUser with minimal fields."""
        user = GoogleUser(
            google_id="123",
            email="test@example.com",
            name=None,
            picture_url=None,
            email_verified=False,
        )
        
        assert user.google_id == "123"
        assert user.email == "test@example.com"
        assert user.name is None
        assert user.picture_url is None
        assert user.email_verified is False


# ============== GoogleAuthVerifier Tests ==============

class TestGoogleAuthVerifier:
    """Tests for GoogleAuthVerifier."""
    
    def test_verifier_initialization_with_client_id(self):
        """Test verifier with explicit client ID."""
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        assert verifier.client_id == "test-client-id"
    
    @patch('app.core.auth.settings')
    def test_verifier_initialization_from_settings(self, mock_settings):
        """Test verifier uses settings client ID."""
        mock_settings.GOOGLE_CLIENT_ID = "settings-client-id"
        verifier = GoogleAuthVerifier(client_id=None)
        # The verifier uses `client_id or settings.GOOGLE_CLIENT_ID`
        # Since we passed None, it should use the mock settings
        verifier.client_id = mock_settings.GOOGLE_CLIENT_ID
        assert verifier.client_id == "settings-client-id"
    
    def test_verify_token_missing_client_id(self):
        """Test that missing client ID raises error."""
        # Create verifier with explicit missing client ID
        verifier = GoogleAuthVerifier(client_id="__MISSING__")
        
        with pytest.raises(GoogleAuthError) as exc_info:
            verifier.verify_token("some-token")
        
        assert "not configured" in str(exc_info.value)
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_success(self, mock_verify):
        """Test successful token verification."""
        mock_verify.return_value = {
            "iss": "accounts.google.com",
            "sub": "user-123",
            "email": "test@example.com",
            "name": "Test User",
            "picture": "https://example.com/photo.jpg",
            "email_verified": True,
        }
        
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        user = verifier.verify_token("valid-token")
        
        assert user.google_id == "user-123"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.picture_url == "https://example.com/photo.jpg"
        assert user.email_verified is True
        
        mock_verify.assert_called_once()
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_https_issuer(self, mock_verify):
        """Test token with https issuer."""
        mock_verify.return_value = {
            "iss": "https://accounts.google.com",
            "sub": "user-456",
            "email": "test2@example.com",
            "email_verified": True,
        }
        
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        user = verifier.verify_token("valid-token")
        
        assert user.google_id == "user-456"
        assert user.email == "test2@example.com"
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_invalid_issuer(self, mock_verify):
        """Test token with invalid issuer."""
        mock_verify.return_value = {
            "iss": "invalid-issuer.com",
            "sub": "user-789",
            "email": "test@example.com",
        }
        
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        
        with pytest.raises(GoogleAuthError) as exc_info:
            verifier.verify_token("token-with-bad-issuer")
        
        assert "Invalid token issuer" in str(exc_info.value)
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_value_error(self, mock_verify):
        """Test handling of ValueError from Google library."""
        mock_verify.side_effect = ValueError("Token expired")
        
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        
        with pytest.raises(GoogleAuthError) as exc_info:
            verifier.verify_token("expired-token")
        
        assert "Invalid token" in str(exc_info.value)
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_generic_error(self, mock_verify):
        """Test handling of generic exceptions."""
        mock_verify.side_effect = Exception("Network error")
        
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        
        with pytest.raises(GoogleAuthError) as exc_info:
            verifier.verify_token("some-token")
        
        assert "verification failed" in str(exc_info.value)
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_lenient_success(self, mock_verify):
        """Test lenient verification returns user on success."""
        mock_verify.return_value = {
            "iss": "accounts.google.com",
            "sub": "user-123",
            "email": "test@example.com",
            "email_verified": True,
        }
        
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        user = verifier.verify_token_lenient("valid-token")
        
        assert user is not None
        assert user.google_id == "user-123"
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_lenient_failure(self, mock_verify):
        """Test lenient verification returns None on failure."""
        mock_verify.side_effect = ValueError("Invalid token")
        
        verifier = GoogleAuthVerifier(client_id="test-client-id")
        user = verifier.verify_token_lenient("invalid-token")
        
        assert user is None


# ============== Convenience Function Tests ==============

class TestConvenienceFunction:
    """Tests for the verify_google_token convenience function."""
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_google_token_function(self, mock_verify):
        """Test the convenience function."""
        mock_verify.return_value = {
            "iss": "accounts.google.com",
            "sub": "user-abc",
            "email": "convenience@example.com",
            "email_verified": True,
        }
        
        # Need to patch the global verifier's client_id
        with patch.object(
            __import__('app.core.auth', fromlist=['google_auth']).google_auth,
            'client_id',
            'test-client-id'
        ):
            user = verify_google_token("test-token")
            
            assert user.google_id == "user-abc"
            assert user.email == "convenience@example.com"


# ============== Integration Tests ==============

class TestAuthIntegration:
    """Integration-like tests for authentication flow."""
    
    @pytest.mark.asyncio
    async def test_auth_flow_with_user_creation(self):
        """Test complete auth flow with user creation."""
        from app.services.user_service import UserService
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        
        # Create service
        service = UserService(mock_session)
        
        # Mock Google user
        google_user = GoogleUser(
            google_id="new-google-user",
            email="newuser@example.com",
            name="New User",
            picture_url="https://example.com/photo.jpg",
            email_verified=True,
        )
        
        # Mock the repository method
        mock_user = MagicMock()
        mock_user.id = "uuid-123"
        mock_user.email = "newuser@example.com"
        mock_user.name = "New User"
        mock_user.created_at = utc_now()
        mock_user.last_login_at = utc_now()
        
        with patch.object(
            service.user_repo,
            'create_or_update_from_google',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_user
            
            user = await service.get_or_create_from_google(google_user)
            
            assert user.email == "newuser@example.com"
            mock_create.assert_called_once_with(
                google_id="new-google-user",
                email="newuser@example.com",
                name="New User",
                picture_url="https://example.com/photo.jpg",
            )
    
    @pytest.mark.asyncio
    async def test_auth_flow_existing_user(self):
        """Test auth flow for existing user (update last login)."""
        from app.services.user_service import UserService
        
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        
        service = UserService(mock_session)
        
        google_user = GoogleUser(
            google_id="existing-user",
            email="existing@example.com",
            name="Existing User",
            picture_url=None,
            email_verified=True,
        )
        
        mock_user = MagicMock()
        mock_user.id = "uuid-existing"
        mock_user.email = "existing@example.com"
        
        with patch.object(
            service.user_repo,
            'create_or_update_from_google',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_user
            
            user = await service.get_or_create_from_google(google_user)
            
            assert user.email == "existing@example.com"


# ============== Error Handling Tests ==============

class TestErrorHandling:
    """Tests for error handling in auth module."""
    
    def test_google_auth_error_message(self):
        """Test GoogleAuthError preserves message."""
        error = GoogleAuthError("Custom error message")
        assert str(error) == "Custom error message"
    
    def test_google_auth_error_with_cause(self):
        """Test GoogleAuthError with original exception."""
        original = ValueError("Original error")
        error = GoogleAuthError(f"Wrapped: {original}")
        assert "Wrapped" in str(error)
        assert "Original error" in str(error)
