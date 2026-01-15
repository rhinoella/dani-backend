from __future__ import annotations

from typing import Dict, List

from app.ingestion.normalizer import normalize_fireflies_transcript
from app.ingestion.chunker import TokenChunker


class IngestionPipeline:
    def __init__(self):
        self.chunker = TokenChunker(
            chunk_size=400,    # Slightly larger chunks for better context
            overlap=100,
            speaker_aware=True
        )

    def process_fireflies_meeting(self, meeting: Dict) -> List[Dict]:
        """
        Process a Fireflies meeting transcript into chunks.
        
        Uses speaker-aware chunking to keep dialogue context together
        and avoid breaking mid-sentence or mid-thought.
        """
        meeting_id = meeting.get('id', '')
        source_file = f"fireflies:{meeting_id}"
        
        # Get meeting metadata
        metadata = {
            "meeting_date": meeting.get("date"),
            "source_file": source_file,
            "title": meeting.get("title", "Unknown meeting"),
            "organizer_email": meeting.get("organizer_email"),
        }
        
        # Get sentences with speaker info
        sentences = meeting.get("sentences", [])
        
        if not sentences:
            # Fallback: use normalized records if no sentences
            normalized = normalize_fireflies_transcript(meeting)
            chunks: List[Dict] = []
            for record in normalized:
                chunks.extend(self.chunker.chunk(record))
            return chunks
        
        # Use speaker-aware chunking for better semantic coherence
        # This groups consecutive utterances by speaker and creates
        # larger, more meaningful chunks
        chunks = self.chunker.chunk_with_speakers(sentences, metadata)
        
        return chunks
