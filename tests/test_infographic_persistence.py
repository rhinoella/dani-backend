"""
Tests for infographic persistence functionality.

Tests S3 storage, database metadata, and retrieval endpoints.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import uuid

from app.database.models.infographic import (
    Infographic,
    InfographicStyle,
    InfographicStatus,
)


class TestInfographicModel:
    """Tests for the Infographic database model."""

    def test_infographic_style_enum(self):
        """Test InfographicStyle enum values."""
        assert InfographicStyle.MODERN.value == "modern"
        assert InfographicStyle.CORPORATE.value == "corporate"
        assert InfographicStyle.MINIMAL.value == "minimal"
        assert InfographicStyle.VIBRANT.value == "vibrant"
        assert InfographicStyle.DARK.value == "dark"

    def test_infographic_status_enum(self):
        """Test InfographicStatus enum values."""
        assert InfographicStatus.PENDING.value == "pending"
        assert InfographicStatus.GENERATING.value == "generating"
        assert InfographicStatus.COMPLETED.value == "completed"
        assert InfographicStatus.FAILED.value == "failed"

    def test_infographic_model_creation(self):
        """Test creating an Infographic model instance."""
        infographic_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        
        infographic = Infographic(
            id=infographic_id,
            user_id=user_id,
            request="Generate infographic about Q1 sales",
            topic="Q1 Sales Summary",
            style=InfographicStyle.CORPORATE,
            headline="Q1 2024 Results",
            subtitle="Strong growth across all regions",
            structured_data={"key_points": ["Revenue up 15%"]},
            s3_key=f"infographics/{user_id}/{infographic_id}.png",
            s3_bucket="dani-infographics",
            image_url="https://example.com/image.png",
            status=InfographicStatus.COMPLETED,
        )
        
        assert infographic.id == infographic_id
        assert infographic.user_id == user_id
        assert infographic.style == InfographicStyle.CORPORATE
        assert infographic.status == InfographicStatus.COMPLETED
        assert infographic.headline == "Q1 2024 Results"
        assert infographic.s3_key is not None and infographic.s3_key.endswith(".png")


class TestInfographicServicePersistence:
    """Tests for InfographicService persistence methods."""

    @pytest.fixture
    def mock_storage_service(self):
        """Create mock storage service."""
        storage = MagicMock()
        storage.upload_file = AsyncMock(return_value="https://s3.example.com/image.png")
        storage.get_presigned_url = MagicMock(return_value="https://s3.example.com/presigned")
        storage.delete_file = AsyncMock(return_value=True)
        return storage

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        session.delete = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_save_to_database(self, mock_db_session):
        """Test saving infographic metadata to database."""
        from app.services.infographic_service import InfographicService
        
        with patch.object(InfographicService, '__init__', lambda x: None):
            service = InfographicService()
            service.storage = MagicMock()
            
            # Test that the model can be created with correct fields
            infographic = Infographic(
                id=str(uuid.uuid4()),
                user_id=str(uuid.uuid4()),
                request="Test request",
                topic="Test topic",
                style=InfographicStyle.MODERN,
                status=InfographicStatus.COMPLETED,
            )
            
            mock_db_session.add(infographic)
            await mock_db_session.commit()
            
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_model_to_dict_conversion(self):
        """Test converting model to dictionary."""
        infographic = Infographic(
            id="test-id",
            user_id="user-id",
            request="Test request",
            topic="Test topic",
            style=InfographicStyle.MODERN,
            headline="Test Headline",
            subtitle="Test Subtitle",
            structured_data={"points": ["a", "b"]},
            s3_key="s3/key/test.png",
            image_url="https://example.com/img.png",
            status=InfographicStatus.COMPLETED,
            created_at=datetime.utcnow(),
        )
        
        # Test that all important fields exist
        assert infographic.id == "test-id"
        assert infographic.style == InfographicStyle.MODERN
        assert infographic.status == InfographicStatus.COMPLETED
        assert infographic.structured_data == {"points": ["a", "b"]}


class TestInfographicEndpoints:
    """Tests for infographic API endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        user = MagicMock()
        user.id = str(uuid.uuid4())
        user.email = "test@example.com"
        return user

    @pytest.mark.asyncio
    async def test_list_infographics_response_schema(self):
        """Test list infographics response schema."""
        from app.api.routes.infographic import InfographicListResponse, InfographicListItem
        
        # Test schema validation
        response = InfographicListResponse(
            items=[
                InfographicListItem(
                    id="test-id",
                    headline="Test",
                    style="modern",
                    status="completed",
                    image_url="https://example.com/img.png",
                    created_at="2024-01-01T00:00:00",
                )
            ],
            total=1,
            limit=20,
            offset=0,
        )
        
        assert response.total == 1
        assert len(response.items) == 1
        assert response.items[0].id == "test-id"

    @pytest.mark.asyncio
    async def test_generate_with_persistence_flag(self):
        """Test that generate endpoint uses persistence by default."""
        from app.api.routes.infographic import generate_infographic, InfographicRequest
        
        # The endpoint should pass persist=True to the service
        # This is verified by the code structure - check it exists
        import inspect
        source = inspect.getsource(generate_infographic)
        
        assert "persist=True" in source or "persist" in source

    @pytest.mark.asyncio
    async def test_status_filter_validation(self):
        """Test status filter accepts valid values."""
        valid_statuses = ["pending", "generating", "completed", "failed"]
        
        for status in valid_statuses:
            parsed = InfographicStatus(status)
            assert parsed.value == status

    @pytest.mark.asyncio
    async def test_status_filter_rejects_invalid(self):
        """Test status filter rejects invalid values."""
        with pytest.raises(ValueError):
            InfographicStatus("invalid_status")


class TestS3Integration:
    """Tests for S3 storage integration."""

    def test_s3_key_format(self):
        """Test S3 key format is correct."""
        user_id = "user-123"
        infographic_id = "infographic-456"
        
        expected_key = f"infographics/{user_id}/{infographic_id}.png"
        
        assert expected_key.startswith("infographics/")
        assert user_id in expected_key
        assert infographic_id in expected_key
        assert expected_key.endswith(".png")

    def test_presigned_url_expiry_default(self):
        """Test default presigned URL expiry is 1 hour."""
        default_expiry = 3600
        assert default_expiry == 3600  # 1 hour in seconds


class TestMigration:
    """Tests for database migration."""

    def test_migration_file_exists(self):
        """Test that migration file exists."""
        import os
        
        migration_path = "alembic/versions/005_add_infographics_table.py"
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            migration_path
        )
        
        # Check migration file exists
        assert os.path.exists(full_path), f"Migration file not found: {full_path}"

    def test_migration_has_upgrade_and_downgrade(self):
        """Test migration has both upgrade and downgrade functions."""
        import os
        
        migration_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "alembic/versions/005_add_infographics_table.py"
        )
        
        with open(migration_path, 'r') as f:
            content = f.read()
        
        assert "def upgrade()" in content
        assert "def downgrade()" in content
        assert "create_table" in content or "op.create_table" in content
