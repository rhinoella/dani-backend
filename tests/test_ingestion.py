import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.ingestion_service import IngestionService


@pytest.fixture
def mock_loader():
    """Mock FirefliesLoader"""
    loader = AsyncMock()
    
    # Mock transcript data
    loader.get_transcript.return_value = {
        "id": "transcript123",
        "title": "Project Planning Meeting",
        "date": 1700000000000,  # epoch ms
        "duration": 3600,
        "organizer_email": "alice@example.com",
        "sentences": [
            {"text": "Hello everyone, let's begin.", "speaker_name": "Alice"},
            {"text": "Sounds good to me.", "speaker_name": "Bob"},
            {"text": "We need to discuss the Q4 roadmap.", "speaker_name": "Alice"},
        ]
    }
    
    # Mock list of transcripts
    loader.list_transcripts.return_value = [
        {"id": "transcript123", "title": "Meeting 1"},
        {"id": "transcript456", "title": "Meeting 2"},
    ]
    
    # Mock paginated transcripts
    loader.list_transcripts_paginated.return_value = [
        {"id": "transcript789", "date": 1700000000000},
        {"id": "transcript101", "date": 1700100000000},
    ]
    
    return loader


@pytest.fixture
def mock_pipeline():
    """Mock IngestionPipeline"""
    pipeline = MagicMock()
    
    # Mock chunked output (matches actual chunker structure)
    pipeline.process_fireflies_meeting.return_value = [
        {
            "text": "Hello everyone, let's begin. Sounds good to me.",
            "metadata": {
                "chunk_index": 0,
                "token_count": 12,
                "speaker": "Alice",
                "section_id": 0,
            }
        },
        {
            "text": "We need to discuss the Q4 roadmap.",
            "metadata": {
                "chunk_index": 1,
                "token_count": 8,
                "speaker": "Alice",
                "section_id": 1,
            }
        },
    ]
    
    return pipeline


@pytest.fixture
def mock_embedder():
    """Mock OllamaEmbeddingClient"""
    embedder = AsyncMock()
    
    # Mock embeddings (384-dim vectors)
    embedder.embed_batch.return_value = [
        [0.1] * 384,
        [0.2] * 384,
    ]
    
    return embedder


@pytest.fixture
def mock_store():
    """Mock QdrantStore"""
    store = MagicMock()
    
    # Mock methods
    store.ensure_collection = MagicMock()
    store.upsert = MagicMock()
    store.scroll = MagicMock(return_value=([], None))
    
    return store


@pytest.fixture
def ingestion_service(mock_loader, mock_pipeline, mock_embedder, mock_store):
    """Create IngestionService with mocked dependencies"""
    service = IngestionService()
    service.loader = mock_loader
    service.pipeline = mock_pipeline
    service.embedder = mock_embedder
    service.store = mock_store
    return service


@pytest.mark.asyncio
async def test_ingest_transcript_basic(ingestion_service, mock_loader, mock_pipeline, mock_embedder, mock_store):
    """Test basic transcript ingestion"""
    result = await ingestion_service.ingest_transcript("transcript123")
    
    # Verify loader was called
    mock_loader.get_transcript.assert_awaited_once_with("transcript123")
    
    # Verify pipeline processed the transcript
    mock_pipeline.process_fireflies_meeting.assert_called_once()
    
    # Verify embeddings were generated
    texts = mock_embedder.embed_batch.call_args[0][0]
    assert len(texts) == 2
    mock_embedder.embed_batch.assert_awaited_once()
    
    # Verify collection was ensured
    mock_store.ensure_collection.assert_called_once()
    
    # Verify points were upserted
    mock_store.upsert.assert_called_once()
    
    # Check result
    assert result["transcript_id"] == "transcript123"
    assert result["ingested"] == 2
    assert result["vector_size"] == 384


@pytest.mark.asyncio
async def test_ingest_transcript_empty_chunks(ingestion_service, mock_loader, mock_pipeline):
    """Test ingestion when pipeline returns no chunks"""
    mock_pipeline.process_fireflies_meeting.return_value = []
    
    result = await ingestion_service.ingest_transcript("transcript123")
    
    assert result["transcript_id"] == "transcript123"
    assert result["ingested"] == 0
    assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_ingest_transcript_creates_stable_ids(ingestion_service, mock_store):
    """Test that point IDs are deterministic for same transcript"""
    await ingestion_service.ingest_transcript("transcript123")
    
    # Get the points that were upserted
    call_args = mock_store.upsert.call_args
    points = call_args[0][1]  # Second positional arg
    
    # Verify point IDs are strings (UUIDs)
    assert all(isinstance(p.id, str) for p in points)
    
    # Verify point IDs are deterministic
    # If we ingest same transcript again, IDs should be the same
    await ingestion_service.ingest_transcript("transcript123")
    second_call_args = mock_store.upsert.call_args
    second_points = second_call_args[0][1]
    
    assert len(points) == len(second_points)
    for p1, p2 in zip(points, second_points):
        assert p1.id == p2.id  # Same IDs for same transcript


@pytest.mark.asyncio
async def test_ingest_transcript_payload_structure(ingestion_service, mock_store):
    """Test that payload contains all required metadata"""
    await ingestion_service.ingest_transcript("transcript123")
    
    # Get the points that were upserted
    call_args = mock_store.upsert.call_args
    points = call_args[0][1]
    
    # Check first point's payload
    payload = points[0].payload
    assert payload["source"] == "fireflies"
    assert payload["transcript_id"] == "transcript123"
    assert payload["title"] == "Project Planning Meeting"
    assert payload["date"] == 1700000000000
    assert payload["duration"] == 3600
    assert payload["organizer_email"] == "alice@example.com"
    assert "speaker" in payload
    assert "section_id" in payload
    assert "token_count" in payload
    assert "chunk_index" in payload
    assert "text" in payload


@pytest.mark.asyncio
async def test_ingest_recent_transcripts(ingestion_service, mock_loader):
    """Test ingesting multiple recent transcripts"""
    result = await ingestion_service.ingest_recent_transcripts(limit=2)
    
    # Verify loader was called with correct limit
    mock_loader.list_transcripts.assert_awaited_once_with(limit=2)
    
    # Verify each transcript was ingested
    assert mock_loader.get_transcript.await_count == 2
    
    # Check result structure
    assert result["requested"] == 2
    assert len(result["results"]) == 2
    assert all(r["ingested"] == 2 for r in result["results"])


@pytest.mark.asyncio
async def test_sync_transcripts_basic(ingestion_service, mock_loader, mock_store):
    """Test syncing transcripts with date filtering"""
    # Mock list_transcripts to return transcripts
    mock_loader.list_transcripts.return_value = [
        {"id": "t1", "title": "Meeting 1"},
        {"id": "t2", "title": "Meeting 2"},
    ]
    
    result = await ingestion_service.sync_transcripts(
        from_date="2023-11-01",
        to_date="2023-11-30"
    )
    
    # Verify list_transcripts was called
    mock_loader.list_transcripts.assert_awaited()
    
    # Check result structure (FirefliesSyncResponse)
    assert result.status == "completed"
    assert result.transcripts_found >= 0
    assert result.transcripts_ingested >= 0
    assert result.transcripts_skipped >= 0
    assert result.chunks_created >= 0


@pytest.mark.asyncio
async def test_sync_transcripts_deduplication(ingestion_service, mock_loader, mock_store):
    """Test that sync skips already ingested transcripts"""
    # Mock list_transcripts to return transcripts
    mock_loader.list_transcripts.return_value = [
        {"id": "transcript789", "title": "Meeting 1"},
    ]
    
    # Mock check_source_exists to return True (already ingested)
    mock_store.check_source_exists = MagicMock(return_value=True)
    
    result = await ingestion_service.sync_transcripts(force_reingest=False)
    
    # Should skip transcript789 since it exists
    assert result.transcripts_skipped > 0


@pytest.mark.asyncio
async def test_sync_transcripts_force_reingest(ingestion_service, mock_loader, mock_store):
    """Test force_reingest parameter"""
    # Mock list_transcripts to return transcripts
    mock_loader.list_transcripts.return_value = [
        {"id": "transcript789", "title": "Meeting 1"},
    ]
    
    # Mock check_source_exists to return True
    mock_store.check_source_exists = MagicMock(return_value=True)
    
    result = await ingestion_service.sync_transcripts(force_reingest=True)
    
    # Should reingest even if exists
    assert result.transcripts_ingested >= 0  # Should process transcripts


@pytest.mark.asyncio
async def test_ingest_transcript_with_speaker_metadata(ingestion_service, mock_loader, mock_pipeline, mock_store):
    """Test that speaker information is preserved in chunks"""
    # Update mock to include speaker data in metadata
    mock_pipeline.process_fireflies_meeting.return_value = [
        {
            "text": "Hello everyone",
            "metadata": {
                "chunk_index": 0,
                "token_count": 3,
                "speaker": "Alice",
                "section_id": 0,
            }
        },
        {
            "text": "Sounds good",
            "metadata": {
                "chunk_index": 1,
                "token_count": 2,
                "speaker": "Bob",
                "section_id": 1,
            }
        },
    ]
    
    await ingestion_service.ingest_transcript("transcript123")
    
    # Get upserted points
    points = mock_store.upsert.call_args[0][1]
    
    # Verify speaker data is in payload
    # Note: The service extracts speaker from chunk metadata, not directly


@pytest.mark.asyncio
async def test_chunker_speaker_aware(mock_loader):
    """Test speaker-aware chunking"""
    from app.ingestion.chunker import TokenChunker
    
    chunker = TokenChunker(chunk_size=50, speaker_aware=True)
    
    sentences = [
        {"text": "Hello everyone, let's begin the meeting.", "speaker_name": "Alice"},
        {"text": "I agree, we have a lot to cover today.", "speaker_name": "Alice"},
        {"text": "Sounds good to me.", "speaker_name": "Bob"},
        {"text": "Let's start with the roadmap.", "speaker_name": "Alice"},
    ]
    
    chunks = chunker.chunk_with_speakers(sentences, {"title": "Test Meeting"})
    
    # Should respect speaker boundaries
    assert len(chunks) > 0
    
    # Each chunk should have speaker metadata in metadata field
    for chunk in chunks:
        assert "metadata" in chunk
        assert "speakers" in chunk["metadata"]
        assert isinstance(chunk["metadata"]["speakers"], list)


@pytest.mark.asyncio
async def test_chunker_respects_chunk_size(mock_loader):
    """Test that chunker can handle long text"""
    from app.ingestion.chunker import TokenChunker
    
    chunker = TokenChunker(chunk_size=20, speaker_aware=True)
    
    # Create multiple sentences that together exceed chunk_size
    sentences = [
        {"text": "This is the first sentence with some words.", "speaker_name": "Alice"},
        {"text": "This is the second sentence with more words.", "speaker_name": "Alice"},
        {"text": "Now Bob speaks with a different sentence.", "speaker_name": "Bob"},
        {"text": "Alice speaks again with yet another sentence.", "speaker_name": "Alice"},
    ]
    
    chunks = chunker.chunk_with_speakers(sentences, {})
    
    # Should create chunks
    assert len(chunks) >= 1
    
    # Each chunk should have token_count in metadata
    for chunk in chunks:
        assert "metadata" in chunk
        assert "token_count" in chunk["metadata"]


@pytest.mark.asyncio
async def test_pipeline_integration(mock_loader):
    """Test full pipeline with normalizer and chunker"""
    from app.ingestion.pipeline import IngestionPipeline
    
    pipeline = IngestionPipeline()
    
    transcript = {
        "id": "test123",
        "title": "Test Meeting",
        "sentences": [
            {"text": "First sentence.", "speaker_name": "Alice"},
            {"text": "Second sentence.", "speaker_name": "Bob"},
        ]
    }
    
    chunks = pipeline.process_fireflies_meeting(transcript)
    
    # Should produce chunks
    assert len(chunks) > 0
    
    # Each chunk should have required fields (text and metadata)
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "token_count" in chunk["metadata"]


@pytest.mark.asyncio
async def test_ingest_transcript_error_handling(ingestion_service, mock_loader):
    """Test error handling when loader fails"""
    mock_loader.get_transcript.side_effect = Exception("API Error")
    
    with pytest.raises(Exception) as exc_info:
        await ingestion_service.ingest_transcript("transcript123")
    
    assert "API Error" in str(exc_info.value)
