from fastapi import APIRouter, Query

from app.ingestion.loaders.fireflies_loader import FirefliesLoader
from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.qdrant import QdrantStore
from app.core.config import settings

router = APIRouter(prefix="/fireflies", tags=["Fireflies"])

loader = FirefliesLoader()
pipeline = IngestionPipeline()
qdrant_store = QdrantStore()


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


@router.get("/stats")
async def get_transcript_stats():
    """
    Get transcript statistics from the vector store.
    
    Returns the total number of transcript chunks ingested in the system.
    """
    try:
        # Get collection info from Qdrant
        collection_info = qdrant_store.client.get_collection(
            collection_name=settings.QDRANT_COLLECTION_TRANSCRIPTS
        )
        
        # Extract points count (number of transcript chunks)
        total_chunks = collection_info.points_count if collection_info else 0
        
        # Get unique transcript count by scrolling with distinct transcript_ids
        # This gives us the number of unique transcripts ingested
        unique_transcripts = set()
        scroll_result = qdrant_store.client.scroll(
            collection_name=settings.QDRANT_COLLECTION_TRANSCRIPTS,
            limit=10000,  # Large limit to get all points
            with_payload=["transcript_id"],
            with_vectors=False,
        )
        
        for point in scroll_result[0]:
            if point.payload and "transcript_id" in point.payload:
                unique_transcripts.add(point.payload["transcript_id"])
        
        return {
            "total_transcripts": len(unique_transcripts),
            "total_chunks": total_chunks,
            "collection": settings.QDRANT_COLLECTION_TRANSCRIPTS,
        }
    except Exception as e:
        # If collection doesn't exist or other error, return zeros
        return {
            "total_transcripts": 0,
            "total_chunks": 0,
            "collection": settings.QDRANT_COLLECTION_TRANSCRIPTS,
            "error": str(e),
        }
