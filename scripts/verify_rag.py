
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.services.retrieval_service import RetrievalService

async def main():
    print("Initializing RetrievalService...")
    service = RetrievalService()
    
    query = "tell me about somatosensory"
    print(f"\nEvaluating query: '{query}'")
    
    try:
        # Search with confidence
        result = await service.search_with_confidence(query, limit=5)
        
        chunks = result["chunks"]
        confidence = result["confidence"]
        
        print(f"\nConfidence: {confidence}")
        print(f"Found {len(chunks)} chunks:")
        
        found_doc = False
        for i, chunk in enumerate(chunks):
            title = chunk.get("title", "Unknown")
            source = chunk.get("search_source", "unknown")
            doc_source = chunk.get("document_source", False)
            score = chunk.get("score", 0.0)
            
            print(f"\n{i+1}. [{score:.4f}] {title} (Source: {source}, IsDoc: {doc_source})")
            print(f"   Excerpt: {chunk.get('text', '')[:100]}...")
            
            if "somatosensory" in title.lower() or "somatosensory" in chunk.get("text", "").lower():
                found_doc = True
        
        if found_doc:
            print("\n✅ SUCCESS: Found relevant document/content regarding 'somatosensory'.")
        else:
            print("\n❌ FAILURE: Did not find 'somatosensory' content in top results.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
