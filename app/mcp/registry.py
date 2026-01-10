"""
MCP Server Registry.

Manages multiple MCP server connections, their lifecycle, and provides
a unified interface for discovering and calling tools across servers.

Features:
- Centralized server management
- Automatic credential injection from config.settings
- Health monitoring and auto-reconnect
- Connection pooling semantics
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPServerNotFoundError,
)
from app.mcp.client import (
    MCPClient,
    extract_all_content,
)
from app.mcp.schemas import (
    MCPRegistryStatus,
    MCPServerConfig,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
)

logger = logging.getLogger(__name__)


class MCPRegistry:
    """
    Registry for managing multiple MCP server connections.

    Provides a unified interface to:
    - Register and manage server configurations
    - Connect/disconnect servers on demand
    - Discover tools across all connected servers
    - Route tool calls to the appropriate server

    Example:
        registry = MCPRegistry()
        registry.register(MCPServerConfig(
            name="nano-banana",
            command="npx",
            args=["nano-banana-mcp"],
            env={"GEMINI_API_KEY": "..."}
        ))
        await registry.connect_all()
        result = await registry.call_tool("nano-banana", "generate_image", {"prompt": "..."})
    """

    def __init__(self):
        """Initialize the registry."""
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._lock = asyncio.Lock()

    @property
    def server_names(self) -> list[str]:
        """Get list of registered server names."""
        return list(self._configs.keys())

    @property
    def connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return [
            name for name, client in self._clients.items() if client.connected
        ]

    def register(self, config: MCPServerConfig) -> None:
        """
        Register an MCP server configuration.

        Args:
            config: Server configuration

        Note:
            This only registers the config. Call connect() to actually connect.
        """
        if config.name in self._configs:
            logger.warning(f"Overwriting existing config for {config.name}")

        self._configs[config.name] = config
        logger.info(f"Registered MCP server: {config.name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister an MCP server.

        Args:
            name: Server name to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name in self._configs:
            del self._configs[name]
            if name in self._clients:
                del self._clients[name]
            logger.info(f"Unregistered MCP server: {name}")
            return True
        return False

    def get_config(self, name: str) -> MCPServerConfig | None:
        """Get server configuration by name."""
        return self._configs.get(name)

    def get_client(self, name: str) -> MCPClient | None:
        """Get connected client by name."""
        client = self._clients.get(name)
        if client and client.connected:
            return client
        return None

    async def connect(self, name: str) -> MCPClient:
        """
        Connect to a specific MCP server.

        Args:
            name: Server name to connect to

        Returns:
            Connected client

        Raises:
            ValueError: If server not registered
            MCPConnectionError: If connection fails
        """
        async with self._lock:
            config = self._configs.get(name)
            if not config:
                raise ValueError(f"Server not registered: {name}")

            if not config.enabled:
                raise ValueError(f"Server is disabled: {name}")

            # Check if already connected
            existing = self._clients.get(name)
            if existing and existing.connected:
                return existing

            # Create new client and connect
            client = MCPClient(config)
            await client.connect()
            self._clients[name] = client

            return client

    async def disconnect(self, name: str) -> None:
        """
        Disconnect from a specific MCP server.

        Args:
            name: Server name to disconnect from
        """
        async with self._lock:
            client = self._clients.get(name)
            if client:
                await client.disconnect()
                del self._clients[name]

    async def connect_all(self, ignore_errors: bool = True) -> dict[str, str | None]:
        """
        Connect to all registered and enabled servers.

        Args:
            ignore_errors: If True, continue connecting other servers on error

        Returns:
            Dict mapping server name to error message (None if successful)
        """
        results: dict[str, str | None] = {}

        for name, config in self._configs.items():
            if not config.enabled:
                results[name] = "disabled"
                continue

            try:
                await self.connect(name)
                results[name] = None
            except Exception as e:
                error = str(e)
                results[name] = error
                logger.error(f"Failed to connect to {name}: {error}")
                if not ignore_errors:
                    raise

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all connected servers."""
        async with self._lock:
            for name in list(self._clients.keys()):
                try:
                    await self._clients[name].disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting from {name}: {e}")
            self._clients.clear()

    def list_all_tools(self) -> dict[str, list[MCPTool]]:
        """
        List all tools from all connected servers.

        Returns:
            Dict mapping server name to list of tools
        """
        tools: dict[str, list[MCPTool]] = {}
        for name, client in self._clients.items():
            if client.connected:
                tools[name] = client.tools
        return tools

    def find_tool(self, tool_name: str) -> tuple[str, MCPTool] | None:
        """
        Find a tool by name across all connected servers.

        Args:
            tool_name: Name of the tool to find

        Returns:
            Tuple of (server_name, tool) or None if not found
        """
        for name, client in self._clients.items():
            if client.connected:
                tool = client.get_tool(tool_name)
                if tool:
                    return (name, tool)
        return None

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
        auto_connect: bool = True,
    ) -> MCPToolResult:
        """
        Call a tool on a specific server.

        Args:
            server_name: Server to call the tool on
            tool_name: Name of the tool
            arguments: Tool arguments
            timeout: Optional timeout
            auto_connect: If True, connect to server if not connected

        Returns:
            Tool result

        Raises:
            ValueError: If server not found
            MCPConnectionError: If not connected and auto_connect is False
            MCPClientError: If tool call fails
        """
        client = self._clients.get(server_name)

        if not client or not client.connected:
            if auto_connect:
                client = await self.connect(server_name)
            else:
                raise MCPConnectionError(f"Not connected to server: {server_name}")

        return await client.call_tool(tool_name, arguments, timeout)

    async def call_tool_auto(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[str, MCPToolResult]:
        """
        Call a tool, automatically finding which server has it.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            timeout: Optional timeout

        Returns:
            Tuple of (server_name, result)

        Raises:
            ValueError: If tool not found on any server
        """
        # Find which server has this tool
        found = self.find_tool(tool_name)
        if not found:
            available = [
                f"{name}: {client.tool_names}"
                for name, client in self._clients.items()
                if client.connected
            ]
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {available}"
            )

        server_name, _ = found
        result = await self.call_tool(server_name, tool_name, arguments, timeout)
        return (server_name, result)

    def get_status(self) -> MCPRegistryStatus:
        """
        Get status of all registered servers.

        Returns:
            Registry status with server details
        """
        servers: list[MCPServerStatus] = []
        total_tools = 0

        for name, config in self._configs.items():
            client = self._clients.get(name)

            if client and client.connected:
                tool_names = client.tool_names
                total_tools += len(tool_names)
                servers.append(
                    MCPServerStatus(
                        name=name,
                        connected=True,
                        tools=tool_names,
                        last_error=client.last_error,
                        connected_at=client.connected_at,
                    )
                )
            else:
                servers.append(
                    MCPServerStatus(
                        name=name,
                        connected=False,
                        tools=[],
                        last_error=client.last_error if client else None,
                        connected_at=None,
                    )
                )

        return MCPRegistryStatus(
            servers=servers,
            total_tools=total_tools,
        )

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all connected servers.

        Returns:
            Dict mapping server name to health status
        """
        health: dict[str, bool] = {}

        for name, client in self._clients.items():
            if client.connected:
                try:
                    # Try to list tools as a health check
                    await client.list_tools()
                    health[name] = True
                except Exception:
                    health[name] = False
            else:
                health[name] = False

        return health


# =============================================================================
# Global Registry Instance
# =============================================================================

# Singleton registry instance
_registry: MCPRegistry | None = None
_initialized: bool = False


def get_registry() -> MCPRegistry:
    """Get the global MCP registry instance."""
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry


async def setup_default_servers(
    gemini_api_key: str | None = None,
    imgbb_api_key: str | None = None,
) -> MCPRegistry:
    """
    Set up the registry with default MCP servers.

    Credentials are retrieved from config.settings if not provided.

    Args:
        gemini_api_key: Google Gemini API key (defaults to settings.GEMINI_API_KEY)
        imgbb_api_key: ImgBB API key (defaults to settings.IMGBB_API_KEY)

    Returns:
        Configured registry (not yet connected)
    """
    from app.mcp.security import credential_manager

    registry = get_registry()

    # Use credentials from settings if not provided
    gemini_key = gemini_api_key or credential_manager.get_gemini_api_key()
    imgbb_key = imgbb_api_key or credential_manager.get_imgbb_api_key()

    # Log credential status (masked)
    if gemini_key:
        masked = credential_manager.mask_credential(gemini_key)
        logger.info(f"Gemini API key configured: {masked}")
    else:
        logger.warning("No Gemini API key configured - image generation unavailable")

    # Register Nano Banana (JavaScript version via npx)
    if gemini_key:
        registry.register(
            MCPServerConfig(
                name="nano-banana",
                command="npx",
                args=["nano-banana-mcp"],
                env={"GEMINI_API_KEY": gemini_key},
                description="AI image generation using Google Gemini 2.5 Flash",
            )
        )
        logger.info("Registered nano-banana MCP server")

    # Register Python version (alternative)
    if gemini_key and imgbb_key:
        registry.register(
            MCPServerConfig(
                name="nano-banana-python",
                command="uvx",
                args=["mcp-nano-banana"],
                env={
                    "GEMINI_API_KEY": gemini_key,
                    "IMGBB_API_KEY": imgbb_key,
                },
                description="AI image generation with ImgBB hosting",
                enabled=False,  # Disabled by default, JS version preferred
            )
        )
        logger.info("Registered nano-banana-python MCP server (disabled)")

    return registry


async def initialize_mcp() -> bool:
    """
    Initialize the MCP subsystem.

    This should be called during application startup. It:
    - Sets up default servers from config
    - Optionally connects to enabled servers

    Returns:
        True if initialization successful, False otherwise
    """
    global _initialized

    if _initialized:
        logger.debug("MCP already initialized")
        return True

    if not settings.MCP_ENABLED:
        logger.info("MCP disabled via settings")
        return False

    try:
        registry = await setup_default_servers()

        # Optionally auto-connect to servers
        if settings.MCP_AUTO_RECONNECT and registry.server_names:
            logger.info("Auto-connecting to MCP servers...")
            results = await registry.connect_all(ignore_errors=True)

            for name, error in results.items():
                if error:
                    logger.warning(f"Failed to connect to {name}: {error}")
                else:
                    logger.info(f"Connected to {name}")

        _initialized = True
        return True

    except Exception as e:
        logger.error(f"MCP initialization failed: {e}")
        return False


async def shutdown_mcp() -> None:
    """
    Shutdown the MCP subsystem.

    Disconnects all servers and cleans up resources.
    """
    global _initialized

    if not _initialized:
        return

    try:
        registry = get_registry()
        await registry.disconnect_all()
        logger.info("MCP shutdown complete")
    except Exception as e:
        logger.error(f"Error during MCP shutdown: {e}")
    finally:
        _initialized = False
