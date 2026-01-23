#!/usr/bin/env python3
"""
Test script for direct Gemini image generation (now integrated into infographic service).
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from app.core.config import settings
from app.services.infographic_service import InfographicService

@pytest.mark.asyncio
async def test_direct_gemini():
    """Test the direct Gemini image generation via infographic service."""
    print("Testing direct Gemini image generation via infographic service...")

    # Check if API key is available via settings
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("❌ GEMINI_API_KEY not found in settings - will use PIL fallback")
    else:
        print(f"✅ GEMINI_API_KEY found in settings (length: {len(api_key)})")

    # Set the environment variable
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    # Create infographic service
    service = InfographicService()

    # Test image generation via the internal method
    prompt = "Create a simple infographic showing 'Hello World' with a blue background and white text."

    try:
        print("Generating test image...")
        image_data = service._generate_image_direct(prompt, width=512, height=384)

        if image_data:
            print("✅ Image generation successful!")
            print(f"Image data size: {len(image_data)} bytes")

            # Save the image for verification
            with open("test_infographic.png", "wb") as f:
                f.write(image_data)
            print("✅ Test image saved as test_infographic.png")

            assert True
        else:
            print("❌ Image generation failed - no data returned")
            assert False, "Image generation failed - no data returned"

    except Exception as e:
        print(f"❌ Image generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Image generation failed with error: {e}"