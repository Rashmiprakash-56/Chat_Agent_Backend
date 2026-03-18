"""
app/schemas/chat.py
Pydantic schemas for chat sessions and messages.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Session schemas
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    title: str = "New Chat"


class SessionRead(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    preview: Optional[str] = None  # last message preview

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    sessions: List[SessionRead]


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

class MessageRead(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    response_type: Optional[str] = None
    agent_used: Optional[str] = None
    trace_url: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageListResponse(BaseModel):
    messages: List[MessageRead]


# ---------------------------------------------------------------------------
# Chat request (extends plain message with optional session_id)
# ---------------------------------------------------------------------------

class ChatStreamRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
