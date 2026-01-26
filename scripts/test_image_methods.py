#!/usr/bin/env python3
"""Test what image generation methods are available."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from app.core.config import settings

def main():
    print("Checking Available Image Generation Methods")
    print("="*60)
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # Get model info for image generation models
    image_models = [
        'gemini-2.0-flash-exp-image-generation',
        'gemini-2.5-flash-image',
        'gemini-3-pro-image-preview',
    ]
    
    for model_name in image_models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        try:
            model = client.models.get(model=model_name)
            print(f"  Name: {model.name}")
            print(f"  Display: {getattr(model, 'display_name', 'N/A')}")
            print(f"  Description: {getattr(model, 'description', 'N/A')[:100]}...")
            
            # Check supported methods
            methods = getattr(model, 'supported_generation_methods', [])
            print(f"  Supported methods: {methods}")
            
            # Check output token limits
            output_limits = getattr(model, 'output_token_limit', 'N/A')
            print(f"  Output token limit: {output_limits}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Try generating with gemini-2.0-flash-exp-image-generation using different approaches
    print(f"\n{'='*60}")
    print("Testing generation approaches")
    print(f"{'='*60}")
    
    test_model = 'gemini-2.0-flash-exp-image-generation'
    prompt = "Draw a simple blue circle"
    
    # Approach 1: generate_content without config
    print(f"\n1. generate_content (no config):")
    try:
        response = client.models.generate_content(
            model=test_model,
            contents=prompt,
        )
        print(f"   Response type: {type(response)}")
        if response.candidates:
            for c in response.candidates:
                if c.content and c.content.parts:
                    for p in c.content.parts:
                        if hasattr(p, 'text'):
                            print(f"   Text: {p.text[:100]}...")
                        if hasattr(p, 'inline_data') and p.inline_data:
                            print(f"   ✅ Got image! MIME: {p.inline_data.mime_type}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Approach 2: generate_content with TEXT+IMAGE modality
    print(f"\n2. generate_content with TEXT modality:")
    try:
        response = client.models.generate_content(
            model=test_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
            )
        )
        print(f"   Response type: {type(response)}")
        if response.candidates:
            for c in response.candidates:
                if c.content and c.content.parts:
                    for p in c.content.parts:
                        if hasattr(p, 'text'):
                            print(f"   Text: {p.text[:200]}...")
                        if hasattr(p, 'inline_data') and p.inline_data:
                            print(f"   ✅ Got image! MIME: {p.inline_data.mime_type}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Approach 3: Check if generate_images works for this model
    print(f"\n3. generate_images API:")
    try:
        response = client.models.generate_images(
            model=test_model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
            )
        )
        print(f"   ✅ Success! Got {len(response.generated_images)} images")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    main()
