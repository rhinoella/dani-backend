"""
Google Authentication module for DANI Engine.

Handles Google ID token verification and user authentication.
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GoogleUser:
    """Verified Google user information."""
    google_id: str  # 'sub' claim
    email: str
    name: Optional[str]
    picture_url: Optional[str]
    email_verified: bool


class GoogleAuthError(Exception):
    """Exception raised for Google authentication errors."""
    pass


class GoogleAuthVerifier:
    """
    Verifies Google ID tokens.
    
    Uses Google's public keys to verify tokens without
    making network calls on every request (keys are cached).
    """
    
    def __init__(self, client_id: Optional[str] = None):
        """
        Initialize verifier with Google Client ID.
        
        Args:
            client_id: Google OAuth Client ID. If not provided,
                      uses settings.GOOGLE_CLIENT_ID
        """
        self.client_id = client_id or settings.GOOGLE_CLIENT_ID
        self._request = google_requests.Request()
    
    def verify_token(self, token: str) -> GoogleUser:
        """
        Verify a Google ID token and extract user information.
        
        Args:
            token: Google ID token from frontend
            
        Returns:
            GoogleUser with verified information
            
        Raises:
            GoogleAuthError: If token is invalid or verification fails
        """
        if not self.client_id or self.client_id == "__MISSING__":
            raise GoogleAuthError(
                "Google Client ID not configured. "
                "Set GOOGLE_CLIENT_ID environment variable."
            )
        
        try:
            # Verify the token
            # This also checks expiration, audience, and issuer
            # clock_skew_in_seconds handles minor time differences between
            # Google's servers and our server (common in Docker containers)
            idinfo = id_token.verify_oauth2_token(
                token,
                self._request,
                self.client_id,
                clock_skew_in_seconds=10,  # Allow 10 seconds of clock drift
            )
            
            # Verify issuer
            if idinfo["iss"] not in ["accounts.google.com", "https://accounts.google.com"]:
                raise GoogleAuthError("Invalid token issuer")
            
            # Extract user information
            return GoogleUser(
                google_id=idinfo["sub"],
                email=idinfo.get("email", ""),
                name=idinfo.get("name"),
                picture_url=idinfo.get("picture"),
                email_verified=idinfo.get("email_verified", False),
            )
            
        except ValueError as e:
            # Token is invalid
            logger.warning(f"Invalid Google token: {e}")
            raise GoogleAuthError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise GoogleAuthError(f"Token verification failed: {str(e)}")
    
    def verify_token_lenient(self, token: str) -> Optional[GoogleUser]:
        """
        Verify token without raising exceptions.
        
        Useful for optional authentication.
        
        Args:
            token: Google ID token
            
        Returns:
            GoogleUser if valid, None otherwise
        """
        try:
            return self.verify_token(token)
        except GoogleAuthError:
            return None


# Global verifier instance
google_auth = GoogleAuthVerifier()


def verify_google_token(token: str) -> GoogleUser:
    """
    Convenience function to verify a Google ID token.
    
    Args:
        token: Google ID token from Authorization header
        
    Returns:
        GoogleUser with verified information
        
    Raises:
        GoogleAuthError: If token is invalid
    """
    return google_auth.verify_token(token)
