"""
Base repository with common CRUD operations.
"""

from typing import TypeVar, Generic, Type, Optional, List, Any
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import uuid
from datetime import datetime, timezone

from app.database.models import Base

ModelType = TypeVar("ModelType", bound=Base)


def utc_now():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class BaseRepository(Generic[ModelType]):
    """
    Base repository with common CRUD operations.
    Supports soft delete by default.
    """
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session
    
    def _apply_soft_delete_filter(self, query: Any) -> Any:
        """Apply soft delete filter if model supports it."""
        if hasattr(self.model, 'deleted_at'):
            query = query.where(getattr(self.model, 'deleted_at').is_(None))
        return query
    
    async def get_by_id(
        self, 
        id: str, 
        include_deleted: bool = False,
        load_relationships: Optional[List[str]] = None
    ) -> Optional[ModelType]:
        """Get entity by ID."""
        query = select(self.model).where(getattr(self.model, 'id') == id)
        
        if not include_deleted:
            query = self._apply_soft_delete_filter(query)
        
        if load_relationships:
            for rel in load_relationships:
                if hasattr(self.model, rel):
                    query = query.options(selectinload(getattr(self.model, rel)))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
        order_by: Optional[str] = None,
        descending: bool = True
    ) -> List[ModelType]:
        """Get all entities with pagination."""
        query = select(self.model)
        
        if not include_deleted:
            query = self._apply_soft_delete_filter(query)
        
        if order_by and hasattr(self.model, order_by):
            column = getattr(self.model, order_by)
            query = query.order_by(column.desc() if descending else column.asc())
        
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count(self, include_deleted: bool = False) -> int:
        """Count all entities."""
        query = select(func.count()).select_from(self.model)
        
        if not include_deleted:
            query = self._apply_soft_delete_filter(query)
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def create(self, **kwargs) -> ModelType:
        """Create a new entity."""
        if 'id' not in kwargs:
            kwargs['id'] = str(uuid.uuid4())
        
        entity = self.model(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, id: str, **kwargs) -> Optional[ModelType]:
        """Update an entity."""
        entity = await self.get_by_id(id)
        if not entity:
            return None
        
        for key, value in kwargs.items():
            if hasattr(entity, key) and value is not None:
                setattr(entity, key, value)
        
        if hasattr(entity, 'updated_at'):
            setattr(entity, 'updated_at', utc_now())
        
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def soft_delete(self, id: str) -> bool:
        """Soft delete an entity."""
        entity = await self.get_by_id(id)
        if not entity:
            return False
        
        if hasattr(entity, 'deleted_at'):
            setattr(entity, 'deleted_at', utc_now())
            await self.session.flush()
            return True
        
        return False
    
    async def hard_delete(self, id: str) -> bool:
        """Permanently delete an entity."""
        entity = await self.get_by_id(id, include_deleted=True)
        if not entity:
            return False
        
        await self.session.delete(entity)
        await self.session.flush()
        return True
    
    async def restore(self, id: str) -> Optional[ModelType]:
        """Restore a soft-deleted entity."""
        entity = await self.get_by_id(id, include_deleted=True)
        if not entity or not hasattr(entity, 'deleted_at'):
            return None
        
        setattr(entity, 'deleted_at', None)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def exists(self, id: str, include_deleted: bool = False) -> bool:
        """Check if entity exists."""
        query = select(func.count()).select_from(self.model).where(getattr(self.model, 'id') == id)
        
        if not include_deleted:
            query = self._apply_soft_delete_filter(query)
        
        result = await self.session.execute(query)
        return (result.scalar() or 0) > 0
    
    async def bulk_create(self, items: List[dict]) -> List[ModelType]:
        """Create multiple entities."""
        entities = []
        for item in items:
            if 'id' not in item:
                item['id'] = str(uuid.uuid4())
            entity = self.model(**item)
            self.session.add(entity)
            entities.append(entity)
        
        await self.session.flush()
        for entity in entities:
            await self.session.refresh(entity)
        return entities
    
    async def bulk_soft_delete(self, ids: List[str]) -> int:
        """Soft delete multiple entities."""
        count = 0
        for id in ids:
            if await self.soft_delete(id):
                count += 1
        return count
