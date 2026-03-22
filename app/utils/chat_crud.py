"""
app/utils/chat_crud.py
Async CRUD helpers for ChatSession and ChatMessage.
"""
from datetime import datetime, timezone
from typing import List, Optional
import uuid

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession
from app.core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

async def create_session(db: AsyncSession, user_id: str, title: str = "New Chat") -> ChatSession:
    session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=title,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    log.info(" Session created: %s (user=%s, title=%s)", session.id, user_id, title)
    return session


async def list_sessions(db: AsyncSession, user_id: str) -> List[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    sessions = list(result.scalars().all())
    log.debug("Listed %d sessions for user=%s", len(sessions), user_id)
    return sessions


async def get_session(db: AsyncSession, session_id: str, user_id: str) -> Optional[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_session(db: AsyncSession, session_id: str, user_id: str) -> bool:
    session = await get_session(db, session_id, user_id)
    if not session:
        log.warning("Delete failed — session not found: %s", session_id)
        return False
    await db.delete(session)
    await db.commit()
    log.info(" Session deleted: %s", session_id)
    return True


async def update_session_title(db: AsyncSession, session_id: str, title: str) -> None:
    session = await db.get(ChatSession, session_id)
    if session:
        session.title = title
        session.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await db.commit()


async def touch_session(db: AsyncSession, session_id: str) -> None:
    """Update updated_at timestamp on a session."""
    session = await db.get(ChatSession, session_id)
    if session:
        session.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await db.commit()


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    response_type: Optional[str] = None,
    agent_used: Optional[str] = None,
    trace_url: Optional[str] = None,
) -> ChatMessage:
    msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        response_type=response_type,
        agent_used=agent_used,
        trace_url=trace_url,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    log.info("Message added: role=%s, session=%s, len=%d", role, session_id, len(content))
    return msg


async def get_messages(
    db: AsyncSession, session_id: str, user_id: str
) -> List[ChatMessage]:
    # Verify session ownership first
    session = await get_session(db, session_id, user_id)
    if not session:
        return []
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return list(result.scalars().all())
