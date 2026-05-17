"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool

from src.config.settings import Settings


def create_engine(settings: Settings, *, for_migrations: bool = False) -> AsyncEngine:
    """Create async SQLAlchemy engine.

    Uses NullPool for Alembic migrations (sync compat) and AsyncAdaptedQueuePool for app use.
    """
    url = settings.vector_store.database_url.replace("postgresql://", "postgresql+asyncpg://")
    return create_async_engine(
        url,
        poolclass=NullPool if for_migrations else AsyncAdaptedQueuePool,
        echo=False,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return an async session factory bound to the given engine."""
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Async generator yielding a session (for dependency injection)."""
    async with session_factory() as session:
        yield session
