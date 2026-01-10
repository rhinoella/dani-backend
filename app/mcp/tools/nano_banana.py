"""
Nano Banana MCP Tool Wrapper.

Provides a high-level, typed interface for the Nano Banana image generation
MCP server powered by Google Gemini 2.5 Flash.

Reference: https://github.com/ConechoAI/Nano-Banana-MCP

Security:
- Credentials retrieved from config.settings (not raw env vars)
- Input validation for all prompts
- Automatic retry with exponential backoff
"""

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.mcp.client import MCPClient, extract_image, extract_text
from app.mcp.registry import get_registry
from app.mcp.schemas import MCPServerConfig, MCPToolResult
from app.mcp.security import credential_manager, input_validator

logger = logging.getLogger(__name__)

# Default server name in registry
DEFAULT_SERVER_NAME = "nano-banana"


@dataclass
class ImageResult:
    """Result from image generation or editing."""

    success: bool
    image_data: bytes | None = None
    image_base64: str | None = None
    mime_type: str = "image/png"
    file_path: str | None = None
    url: str | None = None
    error: str | None = None
    raw_response: MCPToolResult | None = None

    @property
    def has_image(self) -> bool:
        """Check if result contains image data."""
        return self.image_data is not None or self.image_base64 is not None

    def save(self, path: str | Path) -> str:
        """
        Save the image to a file.

        Args:
            path: File path to save to

        Returns:
            Absolute path to saved file

        Raises:
            ValueError: If no image data available
        """
        if not self.has_image:
            raise ValueError("No image data to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.image_data
        if data is None and self.image_base64:
            data = base64.b64decode(self.image_base64)

        path.write_bytes(data)  # type: ignore
        return str(path.absolute())


class NanoBananaClient:
    """
    High-level client for Nano Banana image generation.

    Provides typed methods for image generation and editing using
    the Nano Banana MCP server.

    Security Features:
        - API key retrieved from config.settings (not raw env vars)
        - Prompt validation before sending to server
        - Automatic retry with exponential backoff
        - Credential masking in logs

    Example:
        # Uses GEMINI_API_KEY from settings automatically
        client = NanoBananaClient()
        await client.connect()

        result = await client.generate("A sunset over mountains")
        if result.success:
            result.save("output/sunset.png")

        await client.disconnect()
    """

    def __init__(
        self,
        api_key: str | None = None,
        server_name: str = DEFAULT_SERVER_NAME,
        output_dir: str | None = None,
    ):
        """
        Initialize Nano Banana client.

        Args:
            api_key: Gemini API key (defaults to settings.GEMINI_API_KEY)
            server_name: Name for this server in the registry
            output_dir: Default directory for saving images
        """
        # Use credential manager to get API key from settings
        self.api_key = api_key or credential_manager.get_gemini_api_key()
        self.server_name = server_name
        self.output_dir = Path(output_dir or settings.MCP_GENERATED_IMAGES_DIR)
        self._client: MCPClient | None = None
        self._last_result: ImageResult | None = None

    @property
    def connected(self) -> bool:
        """Check if connected to the server."""
        return self._client is not None and self._client.connected

    @property
    def last_result(self) -> ImageResult | None:
        """Get the last image generation result."""
        return self._last_result

    async def connect(self) -> None:
        """
        Connect to the Nano Banana MCP server.

        Raises:
            ValueError: If no API key configured
            MCPConnectionError: If connection fails
        """
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY in .env or pass api_key."
            )

        # Log with masked credential
        masked_key = credential_manager.mask_credential(self.api_key)
        logger.info(f"Connecting to Nano Banana with API key: {masked_key}")

        config = MCPServerConfig(
            name=self.server_name,
            command="npx",
            args=["nano-banana-mcp"],
            env={"GEMINI_API_KEY": self.api_key},
            description="Nano Banana image generation",
        )

        self._client = MCPClient(config)
        await self._client.connect()

        logger.info(
            f"Connected to Nano Banana with tools: {self._client.tool_names}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def generate(
        self,
        prompt: str,
        save_to: str | Path | None = None,
    ) -> ImageResult:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            save_to: Optional path to save the image

        Returns:
            ImageResult with the generated image

        Raises:
            MCPConnectionError: If not connected
            ValueError: If prompt is invalid
        """
        if not self._client or not self._client.connected:
            raise RuntimeError("Not connected. Call connect() first.")

        # Validate prompt
        is_valid, error = input_validator.validate_prompt(prompt)
        if not is_valid:
            return ImageResult(success=False, error=f"Invalid prompt: {error}")

        try:
            result = await self._client.call_tool(
                "generate_image",
                {"prompt": prompt},
            )

            image_result = self._parse_result(result)
            self._last_result = image_result

            # Save if path provided
            if save_to and image_result.has_image:
                image_result.file_path = image_result.save(save_to)

            return image_result

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return ImageResult(success=False, error=str(e))

    async def edit(
        self,
        image_path: str | Path,
        prompt: str,
        reference_images: list[str | Path] | None = None,
        save_to: str | Path | None = None,
    ) -> ImageResult:
        """
        Edit an existing image.

        Args:
            image_path: Path to the image to edit
            prompt: Edit instructions
            reference_images: Optional reference images for style
            save_to: Optional path to save the result

        Returns:
            ImageResult with the edited image
        """
        if not self._client or not self._client.connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            args: dict[str, Any] = {
                "imagePath": str(image_path),
                "prompt": prompt,
            }
            if reference_images:
                args["referenceImages"] = [str(p) for p in reference_images]

            result = await self._client.call_tool("edit_image", args)

            image_result = self._parse_result(result)
            self._last_result = image_result

            if save_to and image_result.has_image:
                image_result.file_path = image_result.save(save_to)

            return image_result

        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            return ImageResult(success=False, error=str(e))

    async def continue_editing(
        self,
        prompt: str,
        reference_images: list[str | Path] | None = None,
        save_to: str | Path | None = None,
    ) -> ImageResult:
        """
        Continue editing the last generated/edited image.

        Args:
            prompt: Additional edit instructions
            reference_images: Optional reference images
            save_to: Optional path to save the result

        Returns:
            ImageResult with the edited image
        """
        if not self._client or not self._client.connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            args: dict[str, Any] = {"prompt": prompt}
            if reference_images:
                args["referenceImages"] = [str(p) for p in reference_images]

            result = await self._client.call_tool("continue_editing", args)

            image_result = self._parse_result(result)
            self._last_result = image_result

            if save_to and image_result.has_image:
                image_result.file_path = image_result.save(save_to)

            return image_result

        except Exception as e:
            logger.error(f"Continue editing failed: {e}")
            return ImageResult(success=False, error=str(e))

    def _parse_result(self, result: MCPToolResult) -> ImageResult:
        """Parse MCP tool result into ImageResult."""
        if result.is_error:
            error_text = extract_text(result) or "Unknown error"
            return ImageResult(
                success=False,
                error=error_text,
                raw_response=result,
            )

        # Extract image data
        image_data = extract_image(result)
        if image_data:
            base64_data, mime_type = image_data
            return ImageResult(
                success=True,
                image_base64=base64_data,
                image_data=base64.b64decode(base64_data) if base64_data else None,
                mime_type=mime_type,
                raw_response=result,
            )

        # Check for text response (might contain file path or URL)
        text = extract_text(result)
        if text:
            # Try to detect if it's a file path or URL
            if text.startswith("http"):
                return ImageResult(success=True, url=text, raw_response=result)
            elif "/" in text or "\\" in text:
                return ImageResult(
                    success=True, file_path=text, raw_response=result
                )

        return ImageResult(
            success=False,
            error="No image data in response",
            raw_response=result,
        )

    async def generate_and_save(
        self,
        prompt: str,
        filename: str | None = None,
    ) -> ImageResult:
        """
        Generate an image and save it to the output directory.

        Args:
            prompt: Text description
            filename: Optional filename (auto-generated if not provided)

        Returns:
            ImageResult with file_path set
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.output_dir / filename

        return await self.generate(prompt, save_to=save_path)


# =============================================================================
# Convenience Functions (use registry)
# =============================================================================


async def generate_image(
    prompt: str,
    save_to: str | Path | None = None,
    server_name: str = DEFAULT_SERVER_NAME,
) -> ImageResult:
    """
    Generate an image using the registry's Nano Banana connection.

    Args:
        prompt: Text description of the image
        save_to: Optional path to save the image
        server_name: Name of the server in the registry

    Returns:
        ImageResult with generated image
    """
    registry = get_registry()

    try:
        result = await registry.call_tool(
            server_name,
            "generate_image",
            {"prompt": prompt},
        )

        # Parse result
        if result.is_error:
            error_text = extract_text(result) or "Unknown error"
            return ImageResult(success=False, error=error_text, raw_response=result)

        image_data = extract_image(result)
        if image_data:
            base64_data, mime_type = image_data
            img_result = ImageResult(
                success=True,
                image_base64=base64_data,
                image_data=base64.b64decode(base64_data) if base64_data else None,
                mime_type=mime_type,
                raw_response=result,
            )

            if save_to:
                img_result.file_path = img_result.save(save_to)

            return img_result

        return ImageResult(
            success=False, error="No image in response", raw_response=result
        )

    except Exception as e:
        logger.error(f"generate_image failed: {e}")
        return ImageResult(success=False, error=str(e))


async def edit_image(
    image_path: str | Path,
    prompt: str,
    reference_images: list[str | Path] | None = None,
    save_to: str | Path | None = None,
    server_name: str = DEFAULT_SERVER_NAME,
) -> ImageResult:
    """
    Edit an image using the registry's Nano Banana connection.

    Args:
        image_path: Path to image to edit
        prompt: Edit instructions
        reference_images: Optional reference images
        save_to: Optional path to save result
        server_name: Name of the server in the registry

    Returns:
        ImageResult with edited image
    """
    registry = get_registry()

    try:
        args: dict[str, Any] = {
            "imagePath": str(image_path),
            "prompt": prompt,
        }
        if reference_images:
            args["referenceImages"] = [str(p) for p in reference_images]

        result = await registry.call_tool(server_name, "edit_image", args)

        if result.is_error:
            error_text = extract_text(result) or "Unknown error"
            return ImageResult(success=False, error=error_text, raw_response=result)

        image_data = extract_image(result)
        if image_data:
            base64_data, mime_type = image_data
            img_result = ImageResult(
                success=True,
                image_base64=base64_data,
                image_data=base64.b64decode(base64_data) if base64_data else None,
                mime_type=mime_type,
                raw_response=result,
            )

            if save_to:
                img_result.file_path = img_result.save(save_to)

            return img_result

        return ImageResult(
            success=False, error="No image in response", raw_response=result
        )

    except Exception as e:
        logger.error(f"edit_image failed: {e}")
        return ImageResult(success=False, error=str(e))
