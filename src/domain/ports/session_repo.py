"""Session repository port — contract for conversation session persistence adapters."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from src.domain.entities.session import Message, Session


class SessionRepoPort(Protocol):
    """Protocol for conversation session and message persistence adapters."""

    async def get_or_create(self, session_id: UUID) -> Session:
        """Retrieve the session for the given ID, creating it if it does not exist."""
        ...

    async def save_message(self, message: Message) -> None:
        """Persist a single conversation message to the session store."""
        ...

    async def get_history(self, session_id: UUID, limit: int = 20) -> list[Message]:
        """Return the most recent messages for a session, newest last."""
        ...
