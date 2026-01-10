from __future__ import annotations

from typing import Dict, List
from uuid import uuid4


def normalize_fireflies_transcript(
    meeting: Dict,
) -> List[Dict]:
    """
    Convert a Fireflies meeting transcript into a canonical internal format.
    Spec-compliant schema with source_file and section_id.
    """
    meeting_id = meeting['id']
    source_file = f"fireflies:{meeting_id}"
    meeting_date = meeting.get("date")

    normalized = []

    sentences = meeting.get("sentences", [])
    for segment in sentences:
        text = segment.get("text", "").strip()
        if not text:
            continue

        normalized.append(
            {
                "section_id": str(uuid4()),
                "meeting_date": meeting_date,
                "speaker": segment.get("speaker_name") or "Unknown",
                "text": text,
                "source_file": source_file,
            }
        )

    return normalized
