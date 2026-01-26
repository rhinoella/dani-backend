#!/usr/bin/env python3
"""Test Gemini image generation models."""

import sys
import base64
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from app.core.config import settings

def main():
    print("Testing Gemini Image Generation Models")
    print("="*50)
    
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # Models that might support image generation
    models_to_try = [
        'gemini-2.0-flash-exp-image-generation',
        'gemini-2.5-flash-image',
        'gemini-3-pro-image-preview',
    ]
    
    prompt = "Create a professional business infographic image. It should have a blue header with the title 'Key Statistics', and show three stats: Revenue $1.2M, Growth 42%, Users 10K. Use icons and clean modern design."
    
    for model_name in models_to_try:
        print(f"\n{'='*50}")
        print(f"Trying {model_name}...")
        print(f"{'='*50}")
        
        try:
            # For image generation models, use generate_content with image modality
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                )
            )
            
            print(f"Response received!")
            print(f"Response type: {type(response)}")
            
            if response.candidates:
                print(f"Candidates: {len(response.candidates)}")
                for i, candidate in enumerate(response.candidates):
                    print(f"\nCandidate {i}:")
                    if candidate.content and candidate.content.parts:
                        print(f"  Parts: {len(candidate.content.parts)}")
                        for j, part in enumerate(candidate.content.parts):
                            print(f"  Part {j} type: {type(part)}")
                            
                            # Check for inline_data (image)
                            if hasattr(part, 'inline_data') and part.inline_data:
                                print(f"  ✅ Found inline image data!")
                                inline = part.inline_data
                                print(f"    MIME type: {inline.mime_type if hasattr(inline, 'mime_type') else 'unknown'}")
                                
                                # Get the data
                                if hasattr(inline, 'data'):
                                    data = inline.data
                                    if isinstance(data, str):
                                        data = base64.b64decode(data)
                                    
                                    print(f"    Data size: {len(data)} bytes")
                                    
                                    # Save the image
                                    output_path = Path(__file__).parent / f"test_{model_name.replace('.', '_').replace('-', '_')}.png"
                                    output_path.write_bytes(data)
                                    print(f"    ✅ Saved to {output_path}")
                                    return True
                            
                            # Check for text
                            if hasattr(part, 'text') and part.text:
                                print(f"  Text: {part.text[:200]}...")
            else:
                print("No candidates in response")
                print(f"Full response: {response}")
                
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ {type(e).__name__}: {error_msg[:300]}")
            
            # If rate limited, wait
            if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                print("  Waiting 30 seconds for rate limit...")
                time.sleep(30)
    
    return False

if __name__ == "__main__":
    success = main()
    print("\n" + "="*50)
    print("Result:", "SUCCESS" if success else "FAILED")
