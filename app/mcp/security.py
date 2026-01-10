"""
MCP Security Module.

Provides secure credential management, input validation, and security
utilities for MCP operations.
"""

import hashlib
import logging
import re
import secrets
from functools import lru_cache
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Credential Management
# =============================================================================


class MCPCredentialManager:
    """
    Secure credential manager for MCP servers.
    
    - Retrieves credentials from config (not raw env vars)
    - Validates API key formats
    - Masks sensitive data in logs
    - Provides credential fingerprints for debugging
    """

    # Known API key patterns for validation
    _KEY_PATTERNS = {
        "gemini": r"^AI[a-zA-Z0-9_-]{35,}$",  # Google AI keys start with AI
        "imgbb": r"^[a-f0-9]{32}$",  # ImgBB uses 32-char hex
    }

    def __init__(self):
        self._credential_cache: dict[str, str] = {}

    def get_gemini_api_key(self) -> str | None:
        """
        Get the Gemini API key from settings.
        
        Returns:
            API key if configured, None otherwise
        """
        key = settings.GEMINI_API_KEY
        if not key or key == "__MISSING__":
            return None
        return key

    def get_imgbb_api_key(self) -> str | None:
        """
        Get the ImgBB API key from settings.
        
        Returns:
            API key if configured, None otherwise
        """
        key = settings.IMGBB_API_KEY
        if not key or key == "__MISSING__":
            return None
        return key

    def validate_api_key(self, key: str, key_type: str = "generic") -> bool:
        """
        Validate an API key format.
        
        Args:
            key: The API key to validate
            key_type: Type of key (gemini, imgbb, generic)
            
        Returns:
            True if valid format, False otherwise
        """
        if not key or len(key) < 10:
            return False

        pattern = self._KEY_PATTERNS.get(key_type)
        if pattern:
            return bool(re.match(pattern, key))

        # Generic validation - must be alphanumeric with allowed special chars
        return bool(re.match(r"^[a-zA-Z0-9_-]{10,}$", key))

    def mask_credential(self, credential: str, visible_chars: int = 4) -> str:
        """
        Mask a credential for safe logging.
        
        Args:
            credential: The credential to mask
            visible_chars: Number of chars to show at end
            
        Returns:
            Masked string like "***xyz123"
        """
        if not credential:
            return "<empty>"
        if len(credential) <= visible_chars:
            return "***"
        return "***" + credential[-visible_chars:]

    def get_credential_fingerprint(self, credential: str) -> str:
        """
        Get a fingerprint of a credential for debugging.
        
        This allows correlating credentials without exposing them.
        
        Args:
            credential: The credential to fingerprint
            
        Returns:
            Short hash of the credential
        """
        if not credential:
            return "empty"
        return hashlib.sha256(credential.encode()).hexdigest()[:8]

    def get_server_env(self, server_type: str) -> dict[str, str]:
        """
        Get environment variables for a specific MCP server type.
        
        Args:
            server_type: Type of server (nano-banana, nano-banana-python, etc.)
            
        Returns:
            Dict of environment variables
        """
        env: dict[str, str] = {}

        if server_type in ("nano-banana", "nano-banana-python"):
            gemini_key = self.get_gemini_api_key()
            if gemini_key:
                env["GEMINI_API_KEY"] = gemini_key

        if server_type == "nano-banana-python":
            imgbb_key = self.get_imgbb_api_key()
            if imgbb_key:
                env["IMGBB_API_KEY"] = imgbb_key

        return env

    def has_required_credentials(self, server_type: str) -> tuple[bool, str | None]:
        """
        Check if required credentials are available for a server.
        
        Args:
            server_type: Type of server
            
        Returns:
            Tuple of (has_credentials, error_message)
        """
        if server_type in ("nano-banana", "nano-banana-python"):
            if not self.get_gemini_api_key():
                return False, "GEMINI_API_KEY not configured"

        if server_type == "nano-banana-python":
            if not self.get_imgbb_api_key():
                return False, "IMGBB_API_KEY not configured (required for Python MCP)"

        return True, None


# =============================================================================
# Input Validation
# =============================================================================


class MCPInputValidator:
    """
    Validates and sanitizes inputs for MCP operations.
    """

    # Maximum lengths
    MAX_PROMPT_LENGTH = 4000
    MAX_TOOL_NAME_LENGTH = 100
    MAX_ARGUMENT_VALUE_LENGTH = 10000
    MAX_FILE_PATH_LENGTH = 500

    # Blocked patterns in prompts (for safety)
    BLOCKED_PROMPT_PATTERNS = [
        r"<script\b",  # XSS attempts
        r"javascript:",
        r"data:text/html",
    ]

    # Allowed characters in tool names
    TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"

    @classmethod
    def validate_prompt(cls, prompt: str) -> tuple[bool, str | None]:
        """
        Validate an image generation prompt.
        
        Args:
            prompt: The prompt to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt:
            return False, "Prompt is required"

        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            return False, f"Prompt exceeds maximum length of {cls.MAX_PROMPT_LENGTH}"

        # Check for blocked patterns
        prompt_lower = prompt.lower()
        for pattern in cls.BLOCKED_PROMPT_PATTERNS:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return False, "Prompt contains blocked content"

        return True, None

    @classmethod
    def validate_tool_name(cls, name: str) -> tuple[bool, str | None]:
        """
        Validate a tool name.
        
        Args:
            name: Tool name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Tool name is required"

        if len(name) > cls.MAX_TOOL_NAME_LENGTH:
            return False, f"Tool name exceeds maximum length of {cls.MAX_TOOL_NAME_LENGTH}"

        if not re.match(cls.TOOL_NAME_PATTERN, name):
            return False, "Tool name contains invalid characters"

        return True, None

    @classmethod
    def validate_file_path(cls, path: str) -> tuple[bool, str | None]:
        """
        Validate a file path for security.
        
        Args:
            path: File path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not path:
            return False, "File path is required"

        if len(path) > cls.MAX_FILE_PATH_LENGTH:
            return False, f"Path exceeds maximum length of {cls.MAX_FILE_PATH_LENGTH}"

        # Block path traversal
        if ".." in path:
            return False, "Path traversal not allowed"

        # Block absolute paths to system directories
        blocked_prefixes = ["/etc/", "/usr/", "/bin/", "/sbin/", "C:\\Windows"]
        for prefix in blocked_prefixes:
            if path.startswith(prefix):
                return False, f"Access to {prefix} not allowed"

        return True, None

    @classmethod
    def sanitize_arguments(
        cls, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Sanitize tool arguments.
        
        Args:
            arguments: Arguments to sanitize
            
        Returns:
            Tuple of (sanitized_args, warnings)
        """
        sanitized: dict[str, Any] = {}
        warnings: list[str] = []

        for key, value in arguments.items():
            # Validate key
            if not re.match(r"^[a-zA-Z0-9_]+$", key):
                warnings.append(f"Skipping invalid argument key: {key}")
                continue

            # Sanitize value based on type
            if isinstance(value, str):
                if len(value) > cls.MAX_ARGUMENT_VALUE_LENGTH:
                    value = value[: cls.MAX_ARGUMENT_VALUE_LENGTH]
                    warnings.append(f"Truncated argument {key}")

            sanitized[key] = value

        return sanitized, warnings


# =============================================================================
# Command Security
# =============================================================================


class MCPCommandValidator:
    """
    Validates MCP server commands for security.
    """

    @staticmethod
    def get_allowed_commands() -> set[str]:
        """Get the set of allowed subprocess commands."""
        return set(settings.MCP_ALLOWED_COMMANDS.split(","))

    @classmethod
    def validate_command(cls, command: str) -> tuple[bool, str | None]:
        """
        Validate an MCP server command.
        
        Args:
            command: Command to validate (e.g., 'npx', 'uvx')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command:
            return False, "Command is required"

        # Get base command (first word)
        base_cmd = command.split()[0] if " " in command else command

        # Check against allowed list
        allowed = cls.get_allowed_commands()
        if base_cmd not in allowed:
            return False, f"Command '{base_cmd}' not in allowed list: {allowed}"

        # Block shell metacharacters
        dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "{", "}", "<", ">"]
        for char in dangerous_chars:
            if char in command:
                return False, f"Command contains blocked character: {char}"

        return True, None

    @classmethod
    def validate_args(cls, args: list[str]) -> tuple[bool, str | None]:
        """
        Validate command arguments.
        
        Args:
            args: List of arguments
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for arg in args:
            # Block shell metacharacters in args
            dangerous_chars = [";", "&", "|", "`", "$", "(", ")"]
            for char in dangerous_chars:
                if char in arg:
                    return False, f"Argument contains blocked character: {char}"

            # Block obvious injection attempts
            if arg.startswith("-") and "=" in arg:
                # Allow --key=value style args
                continue
            
        return True, None


# =============================================================================
# Singleton Instances
# =============================================================================


@lru_cache(maxsize=1)
def get_credential_manager() -> MCPCredentialManager:
    """Get the singleton credential manager."""
    return MCPCredentialManager()


# Convenience exports
credential_manager = get_credential_manager()
input_validator = MCPInputValidator
command_validator = MCPCommandValidator
