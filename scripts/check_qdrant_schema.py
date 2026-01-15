#!/usr/bin/env python3
"""Check Qdrant vector database structure."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from app.core.config import settings

def check_qdrant():
    client = QdrantClient(url=settings.QDRANT_URL)
    
    # Get all collections
    collections = client.get_collections().collections
    print("=" * 60)
    print("QDRANT VECTOR DATABASE STRUCTURE")
    print("=" * 60)
    print(f"\nCollections: {len(collections)}")
    for c in collections:
        print(f"  - {c.name}")
    
    # For each collection, get details
    for c in collections:
        print(f"\n{'=' * 60}")
        print(f"COLLECTION: {c.name}")
        print("=" * 60)
        
        info = client.get_collection(c.name)
        print(f"  Points count: {info.points_count}")
        print(f"  Vector size: {info.config.params.vectors.size}")
        print(f"  Distance metric: {info.config.params.vectors.distance}")
        
        # Check indexes
        if hasattr(info, 'payload_schema') and info.payload_schema:
            print(f"\n  Payload Indexes:")
            for field, schema in info.payload_schema.items():
                print(f"    - {field}: {schema.data_type}")
        
        # Get a sample point to see payload structure
        if info.points_count > 0:
            sample = client.scroll(
                collection_name=c.name, 
                limit=1, 
                with_payload=True, 
                with_vectors=False
            )
            if sample[0]:
                point = sample[0][0]
                print(f"\n  Sample Payload Fields ({len(point.payload)} fields):")
                for k, v in sorted(point.payload.items()):
                    val_type = type(v).__name__
                    if isinstance(v, str):
                        val_preview = v[:60] + "..." if len(v) > 60 else v
                    elif isinstance(v, list):
                        val_preview = f"[{len(v)} items]"
                    else:
                        val_preview = str(v)[:60]
                    print(f"    {k}: ({val_type}) {val_preview}")

if __name__ == "__main__":
    check_qdrant()
