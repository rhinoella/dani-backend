#!/usr/bin/env python3
"""Test gemini-2.0-flash-exp-image-generation model."""

import sys
import base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
import os
from dotenv import load_dotenv
load_dotenv(override=True)

def main():
    key = os.getenv('GEMINI_API_KEY')
    print(f"API Key: {key[:20]}...")
    
    client = genai.Client(api_key=key)
    
    print("\nTesting gemini-2.0-flash-exp-image-generation...")
    print("="*50)
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp-image-generation',
            contents='Create a simple infographic image with a blue header that says Statistics and shows the number 42%'
        )
        print("Response received!")
        
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            print(f"✅ Got image! MIME: {part.inline_data.mime_type}")
                            data = part.inline_data.data
                            if isinstance(data, str):
                                data = base64.b64decode(data)
                            output = Path(__file__).parent / "gemini_exp_image_test.png"
                            output.write_bytes(data)
                            print(f"✅ Saved to {output} ({len(data)} bytes)")
                            return True
                        if hasattr(part, 'text') and part.text:
                            print(f"Text response: {part.text[:300]}...")
        else:
            print("No candidates")
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)[:400]}")
    
    return False

if __name__ == "__main__":
    success = main()
    print("\n" + "="*50)
    print("Result:", "SUCCESS ✅" if success else "FAILED ❌")
