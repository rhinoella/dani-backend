"""
Tests for Base Repository.
Uses actual SQLAlchemy models for proper testing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import uuid

from app.repositories.base import BaseRepository, utc_now
from app.database.models import User


# ============== Fixtures ==============

@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_user():
    """Create a mock user entity."""
    user = MagicMock(spec=User)
    user.id = str(uuid.uuid4())
    user.email = "test@example.com"
    user.name = "Test User"
    user.deleted_at = None
    user.updated_at = None
    return user


# ============== Tests ==============

class TestUtcNow:
    """Tests for utc_now helper."""
    
    def test_utc_now_returns_datetime(self):
        """Test that utc_now returns datetime."""
        result = utc_now()
        assert isinstance(result, datetime)
    
    def test_utc_now_has_timezone(self):
        """Test that utc_now has UTC timezone."""
        result = utc_now()
        assert result.tzinfo == timezone.utc


class TestBaseRepositoryInit:
    """Tests for BaseRepository initialization."""
    
    def test_init_stores_model_and_session(self, mock_session):
        """Test that init stores model and session."""
        repo = BaseRepository(User, mock_session)
        assert repo.model == User
        assert repo.session == mock_session


class TestBaseRepositoryGetById:
    """Tests for get_by_id method."""
    
    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, mock_user):
        """Test get_by_id returns entity when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_by_id(mock_user.id)
        
        assert result == mock_user
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        """Test get_by_id returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_by_id("nonexistent-id")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_by_id_include_deleted(self, mock_session, mock_user):
        """Test get_by_id with include_deleted flag."""
        mock_user.deleted_at = datetime.now(timezone.utc)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_by_id(mock_user.id, include_deleted=True)
        
        assert result == mock_user


class TestBaseRepositoryGetAll:
    """Tests for get_all method."""
    
    @pytest.mark.asyncio
    async def test_get_all_returns_list(self, mock_session, mock_user):
        """Test get_all returns list of entities."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_user]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_all()
        
        assert result == [mock_user]
    
    @pytest.mark.asyncio
    async def test_get_all_empty(self, mock_session):
        """Test get_all returns empty list when no entities."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_all()
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_all_with_pagination(self, mock_session, mock_user):
        """Test get_all with skip and limit."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_user]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_all(skip=10, limit=5)
        
        assert result == [mock_user]
    
    @pytest.mark.asyncio
    async def test_get_all_with_order_by_descending(self, mock_session, mock_user):
        """Test get_all with descending order."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_user]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_all(order_by="created_at", descending=True)
        
        assert result == [mock_user]
    
    @pytest.mark.asyncio
    async def test_get_all_with_order_by_ascending(self, mock_session, mock_user):
        """Test get_all with ascending order."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_user]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_all(order_by="created_at", descending=False)
        
        assert result == [mock_user]
    
    @pytest.mark.asyncio
    async def test_get_all_include_deleted(self, mock_session, mock_user):
        """Test get_all includes soft-deleted entities when flag set."""
        mock_user.deleted_at = datetime.now(timezone.utc)
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_user]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.get_all(include_deleted=True)
        
        assert result == [mock_user]


class TestBaseRepositoryCount:
    """Tests for count method."""
    
    @pytest.mark.asyncio
    async def test_count_returns_number(self, mock_session):
        """Test count returns entity count."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.count()
        
        assert result == 42
    
    @pytest.mark.asyncio
    async def test_count_returns_zero_on_none(self, mock_session):
        """Test count returns 0 when result is None."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.count()
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_count_include_deleted(self, mock_session):
        """Test count with include_deleted flag."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.count(include_deleted=True)
        
        assert result == 50


class TestBaseRepositoryCreate:
    """Tests for create method."""
    
    @pytest.mark.asyncio
    async def test_create_generates_id(self, mock_session):
        """Test create generates UUID if not provided."""
        mock_session.refresh = AsyncMock()
        
        repo = BaseRepository(User, mock_session)
        
        with patch.object(User, '__init__', return_value=None) as mock_init:
            with patch.object(uuid, 'uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
                result = await repo.create(email="test@example.com")
        
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_uses_provided_id(self, mock_session):
        """Test create uses provided ID."""
        custom_id = str(uuid.uuid4())
        mock_session.refresh = AsyncMock()
        
        repo = BaseRepository(User, mock_session)
        
        with patch.object(User, '__init__', return_value=None):
            result = await repo.create(id=custom_id, email="test@example.com")
        
        mock_session.add.assert_called_once()


class TestBaseRepositoryUpdate:
    """Tests for update method."""
    
    @pytest.mark.asyncio
    async def test_update_success(self, mock_session, mock_user):
        """Test update modifies entity."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.update(mock_user.id, name="Updated Name")
        
        assert result == mock_user
        mock_session.flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_update_not_found(self, mock_session):
        """Test update returns None when entity not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.update("nonexistent-id", name="New Name")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_sets_updated_at(self, mock_session, mock_user):
        """Test update sets updated_at timestamp."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        mock_user.updated_at = None
        
        repo = BaseRepository(User, mock_session)
        
        with patch('app.repositories.base.utc_now') as mock_utc:
            mock_utc.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)
            result = await repo.update(mock_user.id, name="Updated")
        
        # Verify update was called
        mock_session.flush.assert_called()


class TestBaseRepositorySoftDelete:
    """Tests for soft_delete method."""
    
    @pytest.mark.asyncio
    async def test_soft_delete_success(self, mock_session, mock_user):
        """Test soft delete sets deleted_at."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.soft_delete(mock_user.id)
        
        assert result is True
        mock_session.flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_soft_delete_not_found(self, mock_session):
        """Test soft delete returns False when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.soft_delete("nonexistent-id")
        
        assert result is False


class TestBaseRepositoryHardDelete:
    """Tests for hard_delete method."""
    
    @pytest.mark.asyncio
    async def test_hard_delete_success(self, mock_session, mock_user):
        """Test hard delete removes entity."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.hard_delete(mock_user.id)
        
        assert result is True
        mock_session.delete.assert_called_once_with(mock_user)
        mock_session.flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_hard_delete_not_found(self, mock_session):
        """Test hard delete returns False when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.hard_delete("nonexistent-id")
        
        assert result is False


class TestBaseRepositoryRestore:
    """Tests for restore method."""
    
    @pytest.mark.asyncio
    async def test_restore_success(self, mock_session, mock_user):
        """Test restore clears deleted_at."""
        mock_user.deleted_at = datetime.now(timezone.utc)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.restore(mock_user.id)
        
        assert result == mock_user
        mock_session.flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_restore_not_found(self, mock_session):
        """Test restore returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.restore("nonexistent-id")
        
        assert result is None


class TestBaseRepositoryExists:
    """Tests for exists method."""
    
    @pytest.mark.asyncio
    async def test_exists_true(self, mock_session):
        """Test exists returns True when entity exists."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.exists("some-id")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self, mock_session):
        """Test exists returns False when entity doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.exists("nonexistent-id")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists_with_none_result(self, mock_session):
        """Test exists handles None result."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        result = await repo.exists("some-id")
        
        assert result is False


class TestBaseRepositoryBulkCreate:
    """Tests for bulk_create method."""
    
    @pytest.mark.asyncio
    async def test_bulk_create_multiple_items(self, mock_session):
        """Test bulk create creates multiple entities."""
        items = [
            {"email": "user1@example.com"},
            {"email": "user2@example.com"},
            {"email": "user3@example.com"},
        ]
        
        repo = BaseRepository(User, mock_session)
        
        with patch.object(User, '__init__', return_value=None):
            result = await repo.bulk_create(items)
        
        assert mock_session.add.call_count == 3
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_create_with_ids(self, mock_session):
        """Test bulk create preserves provided IDs."""
        custom_id = str(uuid.uuid4())
        items = [
            {"id": custom_id, "email": "user@example.com"},
        ]
        
        repo = BaseRepository(User, mock_session)
        
        with patch.object(User, '__init__', return_value=None):
            result = await repo.bulk_create(items)
        
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_create_empty_list(self, mock_session):
        """Test bulk create with empty list."""
        repo = BaseRepository(User, mock_session)
        result = await repo.bulk_create([])
        
        assert result == []
        mock_session.add.assert_not_called()


class TestBaseRepositoryBulkSoftDelete:
    """Tests for bulk_soft_delete method."""
    
    @pytest.mark.asyncio
    async def test_bulk_soft_delete_all_found(self, mock_session, mock_user):
        """Test bulk soft delete when all entities found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        
        result = await repo.bulk_soft_delete(ids)
        
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_bulk_soft_delete_none_found(self, mock_session):
        """Test bulk soft delete when no entities found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        repo = BaseRepository(User, mock_session)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        
        result = await repo.bulk_soft_delete(ids)
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_bulk_soft_delete_empty_list(self, mock_session):
        """Test bulk soft delete with empty list."""
        repo = BaseRepository(User, mock_session)
        result = await repo.bulk_soft_delete([])
        
        assert result == 0


class TestApplySoftDeleteFilter:
    """Tests for _apply_soft_delete_filter method."""
    
    def test_apply_filter_adds_where_clause(self, mock_session):
        """Test that filter is applied to query."""
        repo = BaseRepository(User, mock_session)
        
        # Create a mock query
        mock_query = MagicMock()
        mock_query.where.return_value = mock_query
        
        result = repo._apply_soft_delete_filter(mock_query)
        
        mock_query.where.assert_called_once()
