from fastapi import APIRouter, Query

from app.ingestion.loaders.fireflies_loader import FirefliesLoader
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/fireflies", tags=["Fireflies"])

loader = FirefliesLoader()
pipeline = IngestionPipeline()


@router.get("/test")
async def test_fireflies():
    return await loader.test_connection()


@router.get("/transcripts")
async def list_transcripts(limit: int = Query(10, ge=1, le=50)):
    return await loader.list_transcripts(limit=limit)


@router.get("/transcript/{transcript_id}")
async def get_transcript(transcript_id: str):
    return await loader.get_transcript(transcript_id)


@router.get("/transcript/{transcript_id}/chunks")
async def get_transcript_chunks(transcript_id: str):
    transcript = await loader.get_transcript(transcript_id)
    chunks = pipeline.process_fireflies_meeting(transcript)

    return {
        "transcript_id": transcript_id,
        "total_chunks": len(chunks),
        "chunks": chunks[:5],  # sample
    }
