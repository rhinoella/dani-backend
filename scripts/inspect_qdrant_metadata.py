import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from qdrant_client import QdrantClient
try:
    from app.core.config import settings
except ImportError:
    # Fallback/mock if imports fail structure dependent
    class settings:
        QDRANT_URL = "http://localhost:6333"
        QDRANT_COLLECTION_TRANSCRIPTS = "meeting_transcripts"

def main():
    print(f"Connecting to {settings.QDRANT_URL}...")
    client = QdrantClient(url=settings.QDRANT_URL)
    
    collection = settings.QDRANT_COLLECTION_TRANSCRIPTS
    print(f"Inspecting collection: {collection}")
    
    try:
        # Scroll 5 points
        result, _ = client.scroll(
            collection_name=collection,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"Found {len(result)} records.")
        for record in result:
            print("-" * 50)
            print(f"ID: {record.id}")
            payload = record.payload
            if not payload:
                print("No payload!")
                continue
                
            # Print keys and values
            if "doc_type" in payload:
                print(f"*** doc_type: {payload['doc_type']} ***")
            else:
                print("*** doc_type MISSING ***")
                
            for k, v in payload.items():
                if k == "text":
                    print(f"{k}: {str(v)[:50]}...")
                else:
                    print(f"{k}: {v}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
