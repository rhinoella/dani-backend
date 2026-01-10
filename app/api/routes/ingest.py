import logging
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks

from app.services.ingestion_service import IngestionService
from app.services.background_ingestion import background_ingestion
from app.schemas.ingest import FirefliesSyncRequest, FirefliesSyncResponse, IngestionStatus

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = logging.getLogger(__name__)

svc = IngestionService()


@router.post("/fireflies/transcript/{transcript_id}")
async def ingest_one_transcript(transcript_id: str):
    """Ingest a single Fireflies transcript by ID."""
    logger.info(f"Ingesting single transcript: {transcript_id}")
    try:
        result = await svc.ingest_transcript(transcript_id)
        return result
    except RuntimeError as e:
        logger.error(f"Ingestion failed for {transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error ingesting {transcript_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ingestion failed unexpectedly")


@router.post("/fireflies/recent")
async def ingest_recent(limit: int = Query(5, ge=1, le=50)):
    """Ingest the most recent Fireflies transcripts."""
    logger.info(f"Ingesting {limit} recent transcripts")
    try:
        return await svc.ingest_recent_transcripts(limit=limit)
    except RuntimeError as e:
        logger.error(f"Recent ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in recent ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ingestion failed unexpectedly")


@router.post("/fireflies/sync", response_model=FirefliesSyncResponse)
async def sync_fireflies(request: FirefliesSyncRequest):
    """
    Sync transcripts from Fireflies with optional date filtering and force re-ingestion.
    
    - **from_date**: Start date (YYYY-MM-DD)
    - **to_date**: End date (YYYY-MM-DD)
    - **force_reingest**: Re-ingest even if transcript already exists
    """
    logger.info(f"Sync request: from={request.from_date}, to={request.to_date}, force={request.force_reingest}")
    try:
        return await svc.sync_transcripts(
            from_date=request.from_date,
            to_date=request.to_date,
            force_reingest=request.force_reingest,
        )
    except RuntimeError as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in sync: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Sync failed unexpectedly")


@router.get("/fireflies/status", response_model=IngestionStatus)
async def get_ingestion_status():
    """
    Get current ingestion status including transcript count and collection info.
    """
    logger.debug("Fetching ingestion status")
    try:
        return await svc.get_status()
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ingestion status")


@router.get("/background/status")
async def get_background_ingestion_status():
    """
    Get background ingestion service status.
    Shows if auto-sync is running and progress information.
    """
    return background_ingestion.get_status()


@router.post("/background/trigger")
async def trigger_background_sync(background_tasks: BackgroundTasks):
    """
    Manually trigger a background sync.
    The sync runs in the background and returns immediately.
    """
    if background_ingestion.progress.sync_in_progress:
        return {
            "status": "already_running",
            "message": "A sync is already in progress",
            "progress": background_ingestion.progress.get_stats(),
        }
    
    # Trigger sync in background
    background_tasks.add_task(background_ingestion._sync_transcripts)
    
    return {
        "status": "started",
        "message": "Background sync triggered",
        "progress": background_ingestion.progress.get_stats(),
    }


@router.post("/background/start")
async def start_background_ingestion():
    """Start the background ingestion service if not already running."""
    if background_ingestion._running:
        return {"status": "already_running", "message": "Background ingestion is already running"}
    
    await background_ingestion.start()
    return {"status": "started", "message": "Background ingestion service started"}


@router.post("/background/stop")
async def stop_background_ingestion():
    """Stop the background ingestion service."""
    if not background_ingestion._running:
        return {"status": "not_running", "message": "Background ingestion is not running"}
    
    await background_ingestion.stop()
    return {"status": "stopped", "message": "Background ingestion service stopped"}
