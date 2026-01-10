#!/usr/bin/env python3
"""
Count Fireflies transcripts.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.loaders.fireflies_loader import FirefliesLoader


async def main():
    loader = FirefliesLoader()

    try:
        transcripts = await loader.list_transcripts(limit=50)
        print(f"📊 Total transcripts: {len(transcripts)}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
