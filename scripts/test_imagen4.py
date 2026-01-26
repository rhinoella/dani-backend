#!/usr/bin/env python3
"""Test Imagen 4.0 image generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from app.core.config import settings

def main():
    print("Testing Imagen 4.0 Image Generation")
    print("="*50)
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    models_to_try = [
        'imagen-4.0-fast-generate-001',
        'imagen-4.0-generate-001',
    ]
    
    for model_name in models_to_try:
        print(f"\nTrying {model_name}...")
        try:
            response = client.models.generate_images(
                model=model_name,
                prompt="A professional business infographic with blue header saying Statistics and showing the number 42%",
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                )
            )
            print(f"  ✅ SUCCESS!")
            
            if response.generated_images:
                img = response.generated_images[0]
                print(f"  Image object: {type(img)}")
                
                if hasattr(img, 'image'):
                    image_obj = img.image
                    data = None
                    
                    if hasattr(image_obj, 'image_bytes'):
                        data = image_obj.image_bytes
                    elif hasattr(image_obj, 'data'):
                        data = image_obj.data
                    else:
                        print(f"  Image attrs: {dir(image_obj)}")
                    
                    if data:
                        print(f"  Data size: {len(data)} bytes")
                        output_path = Path(__file__).parent / f"test_{model_name.replace('.', '_')}.png"
                        output_path.write_bytes(data)
                        print(f"  ✅ Saved to {output_path}")
                        return True
                        
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:200]}")
    
    return False

if __name__ == "__main__":
    success = main()
    print("\n" + "="*50)
    print("Result:", "SUCCESS" if success else "FAILED")
