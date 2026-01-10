from __future__ import annotations

from typing import Dict, List

from app.ingestion.normalizer import normalize_fireflies_transcript
from app.ingestion.chunker import TokenChunker


class IngestionPipeline:
    def __init__(self):
        self.chunker = TokenChunker()

    def process_fireflies_meeting(self, meeting: Dict) -> List[Dict]:
        normalized = normalize_fireflies_transcript(meeting)

        chunks: List[Dict] = []
        for record in normalized:
            chunks.extend(self.chunker.chunk(record))

        return chunks
