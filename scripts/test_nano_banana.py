#!/usr/bin/env python3
"""Test Nano Banana (Gemini) image generation with new API key."""

import sys
import base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from app.core.config import settings

def main():
    print("Testing Nano Banana Image Generation")
    print("="*50)
    print(f"API Key: {settings.GEMINI_API_KEY[:15]}...")
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # Test with gemini-2.5-flash-image (Nano Banana)
    model_name = "gemini-2.5-flash-image"
    prompt = "Create a professional business infographic with a blue header titled 'Key Statistics'. Show three stats: Revenue $1.2M, Growth 42%, Users 10K. Use clean modern design with icons."
    
    print(f"\nModel: {model_name}")
    print(f"Prompt: {prompt[:80]}...")
    print()
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        
        print("Response received!")
        
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Check for image
                        if hasattr(part, 'inline_data') and part.inline_data:
                            print("✅ Got image!")
                            print(f"   MIME type: {part.inline_data.mime_type}")
                            
                            # Get the data
                            data = part.inline_data.data
                            if isinstance(data, str):
                                data = base64.b64decode(data)
                            
                            # Save the image
                            output_path = Path(__file__).parent / "nano_banana_test.png"
                            output_path.write_bytes(data)
                            print(f"   Size: {len(data)} bytes")
                            print(f"   ✅ Saved to {output_path}")
                            return True
                            
                        # Check for text
                        if hasattr(part, 'text') and part.text:
                            print(f"   Text response: {part.text[:300]}...")
        else:
            print("No candidates in response")
            print(f"Full response: {response}")
            
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
    
    return False

if __name__ == "__main__":
    success = main()
    print("\n" + "="*50)
    print("Result:", "SUCCESS ✅" if success else "FAILED ❌")
