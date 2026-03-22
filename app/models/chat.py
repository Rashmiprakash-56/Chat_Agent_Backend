"""
app/models/chat.py
SQLAlchemy models for chat sessions and messages.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship
from fastapi_users_db_sqlalchemy.generics import GUID

from app.utils.database import Base


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # FastAPI-Users User PK is a UUID
    user_id = Column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), nullable=False, default="New Chat")
    created_at = Column(DateTime, default=_now, nullable=False)
    updated_at = Column(DateTime, default=_now, onupdate=_now, nullable=False)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="select",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)          # "user" | "assistant"
    content = Column(Text, nullable=False)              # text content
    response_type = Column(String(10), nullable=True)   # "text" | "code" | None
    agent_used = Column(String(20), nullable=True)      # "rag" | "sql" | "web" | None
    trace_url = Column(String(500), nullable=True)      # LangSmith run URL (assistant only)
    created_at = Column(DateTime, default=_now, nullable=False)

    session = relationship("ChatSession", back_populates="messages")
