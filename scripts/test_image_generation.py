#!/usr/bin/env python3
"""
Test script for Gemini image generation capabilities.

This script tests various approaches to generate images using Google's Gemini API.
"""

import sys
import os
import base64
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings


def test_old_genai_package():
    """Test using the deprecated google.generativeai package."""
    print("\n" + "="*60)
    print("TEST 1: Old google.generativeai package")
    print("="*60)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Check if ImageGenerationModel exists
        if hasattr(genai, 'ImageGenerationModel'):
            print("✅ ImageGenerationModel exists")
            model = genai.ImageGenerationModel("imagen-3.0-generate-002")
            response = model.generate_images(
                prompt="A simple blue square",
                number_of_images=1,
            )
            print(f"✅ Generated: {response}")
            return True
        else:
            print("❌ ImageGenerationModel NOT available in this version")
            print(f"   Available: {[x for x in dir(genai) if not x.startswith('_')]}")
            return False
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        return False


def test_new_genai_client():
    """Test using the new google.genai package."""
    print("\n" + "="*60)
    print("TEST 2: New google.genai Client")
    print("="*60)
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # List available models
        print("Listing available models...")
        models = list(client.models.list())
        image_models = [m for m in models if 'imagen' in m.name.lower() or 'image' in m.name.lower()]
        print(f"Found {len(models)} total models, {len(image_models)} image-related:")
        for m in image_models[:10]:
            print(f"  - {m.name}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        return False


def test_imagen_models():
    """Test Imagen model variants."""
    print("\n" + "="*60)
    print("TEST 3: Imagen Model Variants")
    print("="*60)
    
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # Models to try
    imagen_models = [
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001", 
        "imagen-3.0-fast-generate-001",
        "imagegeneration@006",
        "imagegeneration@005",
    ]
    
    for model_name in imagen_models:
        print(f"\nTrying: {model_name}")
        try:
            response = client.models.generate_images(
                model=model_name,
                prompt="A simple blue infographic with the text 'Hello World'",
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                )
            )
            print(f"  ✅ SUCCESS! Got {len(response.generated_images)} images")
            if response.generated_images:
                img = response.generated_images[0]
                print(f"  Image data type: {type(img)}")
                if hasattr(img, 'image'):
                    print(f"  Image bytes: {len(img.image.image_bytes) if hasattr(img.image, 'image_bytes') else 'N/A'}")
                return model_name, response
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:100]}")
    
    return None, None


def test_gemini_multimodal():
    """Test Gemini models with image generation capabilities."""
    print("\n" + "="*60)
    print("TEST 4: Gemini Multimodal Image Generation")
    print("="*60)
    
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # Try different Gemini models
    models_to_try = [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-exp-image-generation",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    
    for model_name in models_to_try:
        print(f"\nTrying: {model_name}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Generate a simple infographic image with a blue header that says 'Statistics' and shows the number 42",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                )
            )
            print(f"  Response: {response}")
            
            # Check for images in response
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                print(f"  ✅ Found inline image data!")
                                print(f"  MIME type: {part.inline_data.mime_type}")
                                return model_name, response
                            if hasattr(part, 'text'):
                                print(f"  Got text: {part.text[:100]}...")
            
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:100]}")
    
    return None, None


def test_vertex_ai():
    """Test using Vertex AI (requires different setup)."""
    print("\n" + "="*60)
    print("TEST 5: Vertex AI (if configured)")
    print("="*60)
    
    try:
        # Check if we have Vertex AI credentials
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        if not project_id:
            print("⚠️  GOOGLE_CLOUD_PROJECT not set, skipping Vertex AI test")
            return False
        
        from google.cloud import aiplatform
        from vertexai.preview.vision_models import ImageGenerationModel
        
        aiplatform.init(project=project_id, location="us-central1")
        
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        images = model.generate_images(
            prompt="A simple blue infographic",
            number_of_images=1,
        )
        print(f"✅ Vertex AI generated {len(images)} images")
        return True
        
    except ImportError:
        print("⚠️  Vertex AI SDK not installed")
        return False
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        return False


def test_save_generated_image():
    """Test generating and saving an image."""
    print("\n" + "="*60)
    print("TEST 6: Generate and Save Image")
    print("="*60)
    
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # Try with response_modalities for image output
    try:
        print("Trying gemini-2.0-flash-exp with image modality...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Create a visual: A blue rectangle with white text saying 'DANI' in the center",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )
        )
        
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for i, part in enumerate(candidate.content.parts):
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Save the image
                            output_path = Path(__file__).parent / "test_output_image.png"
                            image_data = base64.b64decode(part.inline_data.data) if isinstance(part.inline_data.data, str) else part.inline_data.data
                            output_path.write_bytes(image_data)
                            print(f"✅ Saved image to: {output_path}")
                            print(f"   Size: {len(image_data)} bytes")
                            return str(output_path)
        
        print("❌ No image data in response")
        print(f"   Response: {response}")
        
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
    
    return None


def list_all_models():
    """List all available models."""
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)
    
    from google import genai
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    try:
        models = list(client.models.list())
        print(f"\nTotal models available: {len(models)}\n")
        
        for model in sorted(models, key=lambda m: m.name):
            methods = getattr(model, 'supported_generation_methods', [])
            print(f"  {model.name}")
            if methods:
                print(f"    Methods: {methods}")
                
    except Exception as e:
        print(f"❌ Failed to list models: {e}")


def main():
    print("="*60)
    print("GEMINI IMAGE GENERATION TEST SCRIPT")
    print("="*60)
    print(f"API Key configured: {'Yes' if settings.GEMINI_API_KEY else 'No'}")
    print(f"API Key prefix: {settings.GEMINI_API_KEY[:15]}..." if settings.GEMINI_API_KEY else "None")
    
    # Run tests
    test_old_genai_package()
    test_new_genai_client()
    
    working_model, response = test_imagen_models()
    if working_model:
        print(f"\n🎉 Found working Imagen model: {working_model}")
    
    working_gemini, response = test_gemini_multimodal()
    if working_gemini:
        print(f"\n🎉 Found working Gemini model: {working_gemini}")
    
    test_vertex_ai()
    
    saved_path = test_save_generated_image()
    if saved_path:
        print(f"\n🎉 Successfully generated and saved image: {saved_path}")
    
    # List all available models
    list_all_models()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
