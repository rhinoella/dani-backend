"""
Database connection management for PostgreSQL using async SQLAlchemy.

Provides connection pooling, session management, and lifecycle hooks.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create async engine with connection pooling
async_engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verify connections before using
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session.
    
    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_async_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Alias for FastAPI dependency injection
get_db = get_async_session


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI routes.
    
    Usage:
        async with get_db_context() as db:
            user = await db.get(User, user_id)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database connection and verify connectivity.
    Called on application startup.
    """
    logger.info("Initializing database connection...")
    
    try:
        async with async_engine.begin() as conn:
            # Test connection
            await conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection established successfully")
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise


async def close_db() -> None:
    """
    Close database connections gracefully.
    Called on application shutdown.
    """
    logger.info("Closing database connections...")
    await async_engine.dispose()
    logger.info("✅ Database connections closed")


async def create_tables() -> None:
    """
    Create all tables defined in models.
    For development/testing only - use Alembic migrations in production.
    """
    from app.database.models import Base
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created")


async def drop_tables() -> None:
    """
    Drop all tables. Use with caution!
    For development/testing only.
    """
    from app.database.models import Base
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("⚠️ All database tables dropped")


async def check_health() -> dict:
    """
    Check database health for health endpoint.
    """
    try:
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        return {
            "status": "healthy",
            "database": "postgresql",
            "pool_size": settings.DATABASE_POOL_SIZE,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "postgresql",
            "error": str(e),
        }
