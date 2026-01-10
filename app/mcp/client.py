"""
MCP Client Implementation.

Handles communication with MCP servers via subprocess/stdio transport.
Implements the JSON-RPC 2.0 protocol used by MCP.

Security features:
- Command validation against allowlist
- Input sanitization
- Credential masking in logs
- Rate limiting per server
- Secure subprocess handling
"""

import asyncio
import json
import logging
import os
import time
from asyncio.subprocess import Process
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
)
from app.mcp.schemas import (
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPContent,
    MCPImageContent,
    MCPInitializeResult,
    MCPServerConfig,
    MCPTextContent,
    MCPTool,
    MCPToolResult,
    MCPToolsListResult,
)

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for communicating with MCP servers.

    Spawns the server as a subprocess and communicates via stdio using
    JSON-RPC 2.0 messages.

    Security Features:
        - Command validation against configurable allowlist
        - Input sanitization for all tool calls
        - Credential masking in all log output
        - Rate limiting and concurrent call limits
        - Automatic retry with exponential backoff

    Example:
        config = MCPServerConfig(
            name="nano-banana",
            command="npx",
            args=["nano-banana-mcp"],
            env={"GEMINI_API_KEY": "..."}
        )
        client = MCPClient(config)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("generate_image", {"prompt": "A sunset"})
        await client.disconnect()
    """

    def __init__(
        self,
        config: MCPServerConfig,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
            timeout: Default timeout for operations in seconds
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.config = config
        self.timeout = timeout or settings.MCP_CONNECTION_TIMEOUT
        self.max_retries = max_retries or settings.MCP_MAX_RETRIES
        self.retry_delay = retry_delay or settings.MCP_RETRY_DELAY
        self._process: Process | None = None
        self._request_id = 0
        self._connected = False
        self._connected_at: datetime | None = None
        self._tools: list[MCPTool] = []
        self._last_error: str | None = None
        self._lock = asyncio.Lock()
        self._call_semaphore = asyncio.Semaphore(settings.MCP_MAX_CONCURRENT_CALLS)
        self._call_count = 0
        self._error_count = 0

    @property
    def connected(self) -> bool:
        """Check if connected to the server."""
        return self._connected and self._process is not None

    @property
    def tools(self) -> list[MCPTool]:
        """Get list of available tools."""
        return self._tools

    @property
    def tool_names(self) -> list[str]:
        """Get list of available tool names."""
        return [t.name for t in self._tools]

    @property
    def connected_at(self) -> datetime | None:
        """Get connection timestamp."""
        return self._connected_at

    @property
    def last_error(self) -> str | None:
        """Get last error message."""
        return self._last_error

    def _next_id(self) -> int:
        """Generate next request ID."""
        self._request_id += 1
        return self._request_id

    def _mask_env_for_logging(self, env: dict[str, str]) -> dict[str, str]:
        """Mask sensitive environment variables for logging."""
        masked = {}
        sensitive_keys = {"api_key", "secret", "password", "token", "credential"}
        for key, value in env.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                masked[key] = "***" + value[-4:] if len(value) > 4 else "***"
            else:
                masked[key] = value
        return masked

    def _validate_command(self) -> None:
        """Validate the server command against allowlist."""
        from app.mcp.security import command_validator

        is_valid, error = command_validator.validate_command(self.config.command)
        if not is_valid:
            raise MCPConnectionError(f"Invalid command: {error}")

        is_valid, error = command_validator.validate_args(self.config.args)
        if not is_valid:
            raise MCPConnectionError(f"Invalid arguments: {error}")

    async def connect(self) -> None:
        """
        Connect to the MCP server.

        Spawns the server process and performs initialization handshake.
        Includes command validation, credential masking, and retry logic.

        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If connection times out
        """
        if self._connected:
            logger.debug(f"Already connected to {self.config.name}")
            return

        # Validate command before spawning
        self._validate_command()

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await self._connect_attempt()
                return
            except (MCPConnectionError, MCPTimeoutError) as e:
                last_error = e
                self._error_count += 1
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Connection attempt {attempt + 1}/{self.max_retries} failed for "
                        f"{self.config.name}: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    await self._cleanup()

        # All retries exhausted
        raise MCPConnectionError(
            f"Failed to connect to {self.config.name} after {self.max_retries} attempts: {last_error}"
        )

    async def _connect_attempt(self) -> None:
        """Single connection attempt."""
        try:
            # Build environment (masking sensitive values in logs)
            env = os.environ.copy()
            env.update(self.config.env)
            masked_env = self._mask_env_for_logging(self.config.env)

            # Build command
            cmd = [self.config.command] + self.config.args
            logger.info(
                f"Starting MCP server {self.config.name}: {' '.join(cmd)} "
                f"(env: {masked_env})"
            )

            # Spawn process with security constraints
            # Use larger buffer limit (16MB) for image data responses
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Security: don't inherit shell, limit capabilities
                start_new_session=True,
                limit=16 * 1024 * 1024,  # 16MB buffer for large responses
            )

            # Perform initialization handshake
            await self._initialize()

            # Get available tools
            await self._fetch_tools()

            self._connected = True
            self._connected_at = datetime.now()
            self._last_error = None

            logger.info(
                f"Connected to MCP server {self.config.name} "
                f"with {len(self._tools)} tools: {self.tool_names}"
            )

        except asyncio.TimeoutError as e:
            self._last_error = f"Connection timeout: {e}"
            await self._cleanup()
            raise MCPTimeoutError(self._last_error) from e
        except Exception as e:
            self._last_error = f"Connection failed: {e}"
            await self._cleanup()
            raise MCPConnectionError(self._last_error) from e

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self._connected:
            return

        logger.info(f"Disconnecting from MCP server {self.config.name}")
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up process resources."""
        self._connected = False

        if self._process:
            try:
                # Send shutdown notification (best effort)
                try:
                    notification = JSONRPCNotification(
                        method="notifications/cancelled",
                        params={"reason": "client_shutdown"},
                    )
                    await self._write_message(notification.model_dump())
                except Exception:
                    pass

                # Terminate process
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            finally:
                self._process = None

    async def _initialize(self) -> MCPInitializeResult:
        """
        Perform MCP initialization handshake.

        Returns:
            Initialization result with server capabilities
        """
        # Send initialize request
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "clientInfo": {
                    "name": "DANI-MCP-Client",
                    "version": "1.0.0",
                },
            },
        )

        result = MCPInitializeResult.model_validate(response)

        # Send initialized notification
        notification = JSONRPCNotification(method="notifications/initialized")
        await self._write_message(notification.model_dump())

        return result

    async def _fetch_tools(self) -> None:
        """Fetch available tools from the server."""
        response = await self._send_request("tools/list", {})
        tools_result = MCPToolsListResult.model_validate(response)
        self._tools = tools_result.tools

    async def list_tools(self) -> list[MCPTool]:
        """
        List available tools from the server.

        Returns:
            List of available tools

        Raises:
            MCPConnectionError: If not connected
        """
        if not self.connected:
            raise MCPConnectionError("Not connected to server")

        # Refresh tools list
        await self._fetch_tools()
        return self._tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> MCPToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            timeout: Optional timeout override

        Returns:
            Tool result with content

        Raises:
            MCPConnectionError: If not connected
            MCPToolError: If tool execution fails
            MCPTimeoutError: If operation times out
        """
        from app.mcp.security import input_validator

        if not self.connected:
            raise MCPConnectionError("Not connected to server")

        # Validate tool name
        is_valid, error = input_validator.validate_tool_name(tool_name)
        if not is_valid:
            raise MCPToolError(f"Invalid tool name: {error}")

        # Validate tool exists
        if tool_name not in self.tool_names:
            raise MCPToolError(
                f"Unknown tool: {tool_name}. Available: {self.tool_names}"
            )

        # Sanitize arguments
        sanitized_args = arguments or {}
        if arguments:
            sanitized_args, warnings = input_validator.sanitize_arguments(arguments)
            for warning in warnings:
                logger.warning(f"[{self.config.name}] {warning}")

        start_time = time.time()
        self._call_count += 1

        # Use semaphore to limit concurrent calls
        async with self._call_semaphore:
            try:
                response = await self._send_request(
                    "tools/call",
                    {
                        "name": tool_name,
                        "arguments": sanitized_args,
                    },
                    timeout=timeout or settings.MCP_TOOL_TIMEOUT,
                )

                result = MCPToolResult.model_validate(response)

                elapsed = (time.time() - start_time) * 1000
                logger.info(
                    f"Tool {tool_name} completed in {elapsed:.1f}ms, "
                    f"error={result.is_error}, content_items={len(result.content)}"
                )

                return result

            except asyncio.TimeoutError:
                self._error_count += 1
                self._last_error = f"Tool {tool_name} timed out"
                raise MCPTimeoutError(self._last_error)
            except Exception as e:
                self._error_count += 1
                self._last_error = f"Tool {tool_name} failed: {e}"
                raise MCPToolError(self._last_error) from e

    @property
    def stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "name": self.config.name,
            "connected": self.connected,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "tools": self.tool_names,
        }

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout

        Returns:
            Response result

        Raises:
            MCPError: If request fails
        """
        async with self._lock:
            request = JSONRPCRequest(
                id=self._next_id(),
                method=method,
                params=params,
            )

            await self._write_message(request.model_dump())

            response = await asyncio.wait_for(
                self._read_response(request.id),
                timeout=timeout or self.timeout,
            )

            if response.error:
                error_msg = response.error.get("message", str(response.error))
                raise MCPError(f"RPC error: {error_msg}")

            return response.result

    async def _write_message(self, message: dict[str, Any]) -> None:
        """Write a JSON message to the server."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("No stdin available")

        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

        logger.debug(f"Sent: {message.get('method', 'response')}")

    async def _read_response(self, expected_id: int | str) -> JSONRPCResponse:
        """Read a JSON-RPC response from the server."""
        if not self._process or not self._process.stdout:
            raise MCPConnectionError("No stdout available")

        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise MCPConnectionError("Server closed connection")

            try:
                data = json.loads(line.decode().strip())

                # Skip notifications
                if "id" not in data:
                    logger.debug(f"Received notification: {data.get('method')}")
                    continue

                response = JSONRPCResponse.model_validate(data)

                if response.id == expected_id:
                    return response

                logger.warning(
                    f"Unexpected response id: {response.id}, expected: {expected_id}"
                )

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from server: {e}")
                continue

    def get_tool(self, name: str) -> MCPTool | None:
        """Get a tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def generate_image(self, prompt: str) -> MCPToolResult:
        """
        Generate an image using generate_image tool.

        Args:
            prompt: Text description of the image

        Returns:
            Tool result with image content
        """
        return await self.call_tool("generate_image", {"prompt": prompt})

    async def edit_image(
        self,
        image_path: str,
        prompt: str,
        reference_images: list[str] | None = None,
    ) -> MCPToolResult:
        """
        Edit an existing image.

        Args:
            image_path: Path to image to edit
            prompt: Edit instructions
            reference_images: Optional reference images

        Returns:
            Tool result with edited image
        """
        args: dict[str, Any] = {
            "imagePath": image_path,
            "prompt": prompt,
        }
        if reference_images:
            args["referenceImages"] = reference_images

        return await self.call_tool("edit_image", args)

    async def continue_editing(
        self,
        prompt: str,
        reference_images: list[str] | None = None,
    ) -> MCPToolResult:
        """
        Continue editing the last generated/edited image.

        Args:
            prompt: Additional edit instructions
            reference_images: Optional reference images

        Returns:
            Tool result with edited image
        """
        args: dict[str, Any] = {"prompt": prompt}
        if reference_images:
            args["referenceImages"] = reference_images

        return await self.call_tool("continue_editing", args)


# =============================================================================
# Helper Functions
# =============================================================================


def extract_text(result: MCPToolResult) -> str | None:
    """Extract text content from a tool result."""
    for content in result.content:
        if isinstance(content, MCPTextContent):
            return content.text
        if isinstance(content, dict) and content.get("type") == "text":
            return content.get("text")
    return None


def extract_image(result: MCPToolResult) -> tuple[str, str] | None:
    """
    Extract image content from a tool result.

    Returns:
        Tuple of (base64_data, mime_type) or None
    """
    for content in result.content:
        if isinstance(content, MCPImageContent):
            return (content.data, content.mime_type)
        if isinstance(content, dict) and content.get("type") == "image":
            return (content.get("data", ""), content.get("mimeType", "image/png"))
    return None


def extract_all_content(result: MCPToolResult) -> list[dict[str, Any]]:
    """Extract all content as dictionaries."""
    contents = []
    for content in result.content:
        if isinstance(content, (MCPTextContent, MCPImageContent)):
            contents.append(content.model_dump())
        elif isinstance(content, dict):
            contents.append(content)
    return contents
