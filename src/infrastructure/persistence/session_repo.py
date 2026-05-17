"""Session repository — SQLAlchemy 2.0 async adapter implementing SessionRepoPort."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from ulid import ULID

from src.domain.entities.session import Message, MessageRole, Session
from src.infrastructure.persistence.models import MessageORM, SessionORM
from src.shared.metrics import inc_counter

__all__ = ["SessionRepository"]

AsyncSessionFactory = async_sessionmaker[AsyncSession]
_METRIC_MESSAGES_APPENDED = "session_messages_appended_total"


class SessionRepository:
    """PostgreSQL-backed session and message persistence adapter.

    Implements the ``SessionRepoPort`` contract using SQLAlchemy 2.0
    async sessions.  ULID strings are used as the public session
    identifier (time-sortable, 26-char), stored as UUID primary keys
    in PostgreSQL via ``ULID.to_uuid()``.

    Args:
        session_factory: Async session factory produced by
            :func:`~src.infrastructure.persistence.engine.create_session_factory`.
    """

    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        """Store the async session factory for later use.

        Args:
            session_factory: Callable that opens an async DB session.
        """
        self._session_factory = session_factory

    async def create_session(self, user_id: str | None = None) -> str:
        """Create a new chat session and return its ULID string identifier.

        A fresh :class:`~src.infrastructure.persistence.models.SessionORM` row
        is inserted with the ULID value converted to a UUID for the primary key.

        Args:
            user_id: Optional user identifier to associate with the session.

        Returns:
            26-character ULID string (e.g. ``01KRNJ2MNPN1DMJPFNQV41K4SZ``).
        """
        ulid_obj = ULID()
        session_uuid = ulid_obj.to_uuid()
        now = datetime.now(UTC)
        orm = SessionORM(
            id=session_uuid,
            user_id=user_id,
            created_at=now,
            last_active=now,
        )
        async with self._session_factory() as db:
            db.add(orm)
            await db.commit()
        return str(ulid_obj)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to an existing session.

        Creates a :class:`~src.infrastructure.persistence.models.MessageORM`
        row and bumps ``SessionORM.last_active`` atomically within a single
        transaction.  The ``metadata`` parameter is accepted for API
        compatibility but not persisted in the current schema.

        Args:
            session_id: 26-char ULID string identifying the parent session.
            role: Speaker role — one of ``user``, ``assistant``, ``system``.
            content: Message text content (must be non-empty).
            metadata: Optional extra key-value data (currently ignored).
        """
        session_uuid = ULID.from_str(session_id).to_uuid()
        now = datetime.now(UTC)
        msg_orm = MessageORM(
            id=uuid.uuid4(),
            session_id=session_uuid,
            role=role,
            content=content,
            created_at=now,
        )
        async with self._session_factory() as db:
            db.add(msg_orm)
            await db.execute(
                update(SessionORM).where(SessionORM.id == session_uuid).values(last_active=now)
            )
            await db.commit()
        inc_counter(_METRIC_MESSAGES_APPENDED, {"role": role})

    async def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Fetch messages for a session ordered by creation time ascending.

        Args:
            session_id: 26-char ULID string identifying the session.
            limit: Optional cap on number of messages returned.

        Returns:
            List of :class:`~src.domain.entities.session.Message` domain
            entities ordered oldest-first.
        """
        session_uuid = ULID.from_str(session_id).to_uuid()
        stmt = (
            select(MessageORM)
            .where(MessageORM.session_id == session_uuid)
            .order_by(MessageORM.created_at.asc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            rows = result.scalars().all()
        return [
            Message(
                id=row.id,
                session_id=row.session_id,
                role=MessageRole(row.role),
                content=row.content,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def list_sessions(
        self,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List session summaries, optionally filtered by user.

        Args:
            user_id: Optional user identifier to filter results.

        Returns:
            List of dicts with keys ``id`` (ULID string), ``user_id``,
            ``created_at``, ``last_active``, ``message_count``.
        """
        stmt = (
            select(
                SessionORM.id,
                SessionORM.user_id,
                SessionORM.created_at,
                SessionORM.last_active,
                func.count(MessageORM.id).label("message_count"),
            )
            .outerjoin(MessageORM, MessageORM.session_id == SessionORM.id)
            .group_by(SessionORM.id)
            .order_by(SessionORM.last_active.desc())
        )
        if user_id is not None:
            stmt = stmt.where(SessionORM.user_id == user_id)
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            rows = result.all()
        return [
            {
                "id": str(ULID.from_uuid(row.id)),
                "user_id": row.user_id,
                "created_at": row.created_at,
                "last_active": row.last_active,
                "message_count": row.message_count,
            }
            for row in rows
        ]

    async def get_or_create(self, session_id: uuid.UUID) -> Session:
        """Retrieve an existing session or create one for the given UUID.

        Args:
            session_id: UUID identifying the session (maps 1-to-1 to the PK).

        Returns:
            :class:`~src.domain.entities.session.Session` domain entity.
        """
        async with self._session_factory() as db:
            result = await db.execute(select(SessionORM).where(SessionORM.id == session_id))
            orm = result.scalar_one_or_none()
            if orm is None:
                now = datetime.now(UTC)
                orm = SessionORM(
                    id=session_id,
                    created_at=now,
                    last_active=now,
                )
                db.add(orm)
                await db.commit()
                await db.refresh(orm)
        return Session(
            id=orm.id,
            user_id=orm.user_id,
            created_at=orm.created_at,
            last_active=orm.last_active,
        )

    async def save_message(self, message: Message) -> None:
        """Persist a domain :class:`~src.domain.entities.session.Message`.

        Also bumps ``SessionORM.last_active`` for the parent session.

        Args:
            message: Fully-constructed domain Message entity to persist.
        """
        now = datetime.now(UTC)
        msg_orm = MessageORM(
            id=message.id,
            session_id=message.session_id,
            role=message.role.value,
            content=message.content,
            created_at=message.created_at,
        )
        async with self._session_factory() as db:
            db.add(msg_orm)
            await db.execute(
                update(SessionORM)
                .where(SessionORM.id == message.session_id)
                .values(last_active=now)
            )
            await db.commit()
        inc_counter(_METRIC_MESSAGES_APPENDED, {"role": message.role.value})
