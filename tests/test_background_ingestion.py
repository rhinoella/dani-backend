"""
Tests for Background Ingestion Service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime
from pathlib import Path
import json
import tempfile
import os

from app.services.background_ingestion import (
    BackgroundIngestionService,
    IngestionProgress,
    _stable_point_id,
    PROGRESS_FILE,
)


# ============== Tests ==============

class TestStablePointId:
    """Tests for _stable_point_id helper."""
    
    def test_stable_point_id_single_part(self):
        """Test stable ID generation with single part."""
        result = _stable_point_id("test")
        assert isinstance(result, str)
        assert len(result) == 32  # UUID hex without dashes
    
    def test_stable_point_id_multiple_parts(self):
        """Test stable ID generation with multiple parts."""
        result = _stable_point_id("part1", "part2", "part3")
        assert isinstance(result, str)
        assert len(result) == 32
    
    def test_stable_point_id_deterministic(self):
        """Test that same input produces same output."""
        result1 = _stable_point_id("test", "parts")
        result2 = _stable_point_id("test", "parts")
        assert result1 == result2
    
    def test_stable_point_id_different_inputs(self):
        """Test that different inputs produce different outputs."""
        result1 = _stable_point_id("test1")
        result2 = _stable_point_id("test2")
        assert result1 != result2


class TestIngestionProgress:
    """Tests for IngestionProgress class."""
    
    @pytest.fixture
    def temp_progress_file(self):
        """Create a temporary progress file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "last_sync": "2024-12-15T10:00:00",
                "total_ingested": 50,
                "total_skipped": 5,
                "ingested_ids": ["id1", "id2", "id3"],
                "last_error": None,
                "sync_in_progress": False,
            }, f)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)
    
    @pytest.fixture
    def new_progress_file(self):
        """Create path for new progress file."""
        temp_path = Path(tempfile.mkdtemp()) / "progress.json"
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)
        if temp_path.parent.exists():
            os.rmdir(temp_path.parent)
    
    def test_init_creates_data_dir(self, new_progress_file):
        """Test that init creates data directory."""
        progress = IngestionProgress(progress_file=new_progress_file)
        assert new_progress_file.parent.exists()
    
    def test_load_existing_file(self, temp_progress_file):
        """Test loading existing progress file."""
        progress = IngestionProgress(progress_file=temp_progress_file)
        
        assert progress.last_sync == "2024-12-15T10:00:00"
        assert progress.total_ingested == 50
        assert "id1" in progress.ingested_ids
    
    def test_load_new_file(self, new_progress_file):
        """Test loading when file doesn't exist."""
        progress = IngestionProgress(progress_file=new_progress_file)
        
        assert progress.last_sync is None
        assert progress.total_ingested == 0
        assert len(progress.ingested_ids) == 0
    
    def test_load_corrupted_file(self, new_progress_file):
        """Test loading corrupted progress file."""
        # Create corrupted file
        new_progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(new_progress_file, 'w') as f:
            f.write("not valid json{")
        
        progress = IngestionProgress(progress_file=new_progress_file)
        
        # Should fall back to defaults
        assert progress.total_ingested == 0
    
    def test_mark_ingested(self, new_progress_file):
        """Test marking transcript as ingested."""
        progress = IngestionProgress(progress_file=new_progress_file)
        
        progress.mark_ingested("transcript-123")
        
        assert "transcript-123" in progress.ingested_ids
        assert progress.total_ingested == 1
    
    def test_mark_ingested_duplicate(self, new_progress_file):
        """Test marking same transcript twice."""
        progress = IngestionProgress(progress_file=new_progress_file)
        
        progress.mark_ingested("transcript-123")
        progress.mark_ingested("transcript-123")
        
        # Should not count twice
        assert progress.total_ingested == 1
    
    def test_mark_skipped(self, new_progress_file):
        """Test marking transcript as skipped."""
        progress = IngestionProgress(progress_file=new_progress_file)
        
        progress.mark_skipped()
        progress.mark_skipped()
        
        assert progress._data["total_skipped"] == 2
    
    def test_start_sync(self, new_progress_file):
        """Test starting sync."""
        progress = IngestionProgress(progress_file=new_progress_file)
        
        progress.start_sync()
        
        assert progress.sync_in_progress is True
    
    def test_end_sync_success(self, new_progress_file):
        """Test ending sync successfully."""
        progress = IngestionProgress(progress_file=new_progress_file)
        progress.start_sync()
        
        progress.end_sync()
        
        assert progress.sync_in_progress is False
        assert progress.last_sync is not None
        assert progress._data["last_error"] is None
    
    def test_end_sync_with_error(self, new_progress_file):
        """Test ending sync with error."""
        progress = IngestionProgress(progress_file=new_progress_file)
        progress.start_sync()
        
        progress.end_sync(error="Connection failed")
        
        assert progress.sync_in_progress is False
        assert progress._data["last_error"] == "Connection failed"
    
    def test_get_stats(self, temp_progress_file):
        """Test getting statistics."""
        progress = IngestionProgress(progress_file=temp_progress_file)
        
        stats = progress.get_stats()
        
        assert stats["last_sync"] == "2024-12-15T10:00:00"
        assert stats["total_ingested"] == 50
        assert stats["total_skipped"] == 5
        assert stats["unique_transcripts"] == 3
        assert stats["sync_in_progress"] is False
    
    def test_save_failure_logged(self, new_progress_file):
        """Test that save failure is logged."""
        progress = IngestionProgress(progress_file=new_progress_file)
        
        # Make directory read-only to cause save failure
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Should not raise, just log
            progress.mark_ingested("test")


class TestBackgroundIngestionService:
    """Tests for BackgroundIngestionService."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies."""
        with patch('app.services.background_ingestion.FirefliesLoader') as mock_loader, \
             patch('app.services.background_ingestion.OllamaEmbeddingClient') as mock_embedder, \
             patch('app.services.background_ingestion.QdrantStore') as mock_store, \
             patch('app.services.background_ingestion.TokenChunker') as mock_chunker, \
             patch('app.services.background_ingestion.IngestionProgress') as mock_progress:
            
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance
            
            mock_embedder_instance = AsyncMock()
            mock_embedder.return_value = mock_embedder_instance
            
            mock_store_instance = MagicMock()
            mock_store.return_value = mock_store_instance
            
            mock_chunker_instance = MagicMock()
            mock_chunker.return_value = mock_chunker_instance
            
            mock_progress_instance = MagicMock()
            mock_progress_instance.ingested_ids = set()
            mock_progress_instance.sync_in_progress = False
            mock_progress.return_value = mock_progress_instance
            
            yield {
                'loader': mock_loader_instance,
                'embedder': mock_embedder_instance,
                'store': mock_store_instance,
                'chunker': mock_chunker_instance,
                'progress': mock_progress_instance,
            }
    
    def test_init(self, mock_dependencies):
        """Test service initialization."""
        service = BackgroundIngestionService()
        
        assert service._running is False
        assert service._task is None
    
    @pytest.mark.asyncio
    async def test_start(self, mock_dependencies):
        """Test starting background ingestion."""
        service = BackgroundIngestionService()
        service._run_sync_loop = AsyncMock()
        
        await service.start()
        
        assert service._running is True
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, mock_dependencies):
        """Test start when already running."""
        service = BackgroundIngestionService()
        service._running = True
        
        await service.start()
        
        # Should not start again
        assert service._task is None
    
    @pytest.mark.asyncio
    async def test_stop(self, mock_dependencies):
        """Test stopping background ingestion."""
        import asyncio
        
        service = BackgroundIngestionService()
        service._running = True
        
        # Create a real asyncio task that we can cancel
        async def dummy_task():
            await asyncio.sleep(10)  # Long sleep that will be cancelled
        
        service._task = asyncio.create_task(dummy_task())
        
        await service.stop()
        
        assert service._running is False
        assert service._task.cancelled()
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, mock_dependencies):
        """Test stop when not running."""
        service = BackgroundIngestionService()
        service._running = False
        
        await service.stop()
        
        # Should not raise
        assert service._running is False
    
    def test_get_status(self, mock_dependencies):
        """Test getting service status."""
        service = BackgroundIngestionService()
        service._running = True
        mock_dependencies['progress'].get_stats.return_value = {
            "total_ingested": 100,
            "sync_in_progress": False,
        }
        
        status = service.get_status()
        
        assert status["running"] is True
        assert status["progress"]["total_ingested"] == 100


class TestIngestionProgressProperties:
    """Tests for IngestionProgress property accessors."""
    
    @pytest.fixture
    def new_progress_file(self):
        """Create path for new progress file."""
        temp_path = Path(tempfile.mkdtemp()) / "progress.json"
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)
        if temp_path.parent.exists():
            os.rmdir(temp_path.parent)
    
    def test_last_sync_property(self, new_progress_file):
        """Test last_sync property."""
        progress = IngestionProgress(progress_file=new_progress_file)
        assert progress.last_sync is None
        
        progress._data["last_sync"] = "2024-12-15"
        assert progress.last_sync == "2024-12-15"
    
    def test_total_ingested_property(self, new_progress_file):
        """Test total_ingested property."""
        progress = IngestionProgress(progress_file=new_progress_file)
        assert progress.total_ingested == 0
        
        progress._data["total_ingested"] = 42
        assert progress.total_ingested == 42
    
    def test_ingested_ids_property(self, new_progress_file):
        """Test ingested_ids property returns set."""
        progress = IngestionProgress(progress_file=new_progress_file)
        assert isinstance(progress.ingested_ids, set)
        assert len(progress.ingested_ids) == 0
        
        progress._data["ingested_ids"] = ["a", "b", "c"]
        assert progress.ingested_ids == {"a", "b", "c"}
    
    def test_sync_in_progress_property(self, new_progress_file):
        """Test sync_in_progress property."""
        progress = IngestionProgress(progress_file=new_progress_file)
        assert progress.sync_in_progress is False
        
        progress._data["sync_in_progress"] = True
        assert progress.sync_in_progress is True
