"""
MCP (Model Context Protocol) Client Module.

This module provides infrastructure for DANI to connect to external MCP servers
and use their tools (image generation, data visualization, etc.).

Security Features:
- Credential management via config.settings
- Command validation against allowlist
- Input sanitization for all tool calls
- Automatic retry with exponential backoff
"""

from app.core.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPServerNotFoundError,
    MCPTimeoutError,
    MCPToolError,
    MCPToolNotFoundError,
)
from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry, initialize_mcp, shutdown_mcp
from app.mcp.schemas import (
    MCPServerConfig,
    MCPTool,
    MCPToolCall,
    MCPToolResult,
)
from app.mcp.security import (
    MCPCredentialManager,
    MCPInputValidator,
    MCPCommandValidator,
    credential_manager,
)

__all__ = [
    # Client
    "MCPClient",
    "MCPRegistry",
    # Lifecycle
    "initialize_mcp",
    "shutdown_mcp",
    # Schemas
    "MCPServerConfig",
    "MCPTool",
    "MCPToolCall",
    "MCPToolResult",
    # Security
    "MCPCredentialManager",
    "MCPInputValidator",
    "MCPCommandValidator",
    "credential_manager",
    # Exceptions
    "MCPError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPToolError",
    "MCPServerNotFoundError",
    "MCPToolNotFoundError",
]
