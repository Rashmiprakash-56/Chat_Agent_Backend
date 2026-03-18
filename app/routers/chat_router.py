"""
app/routers/chat_router.py

Chat endpoints with session persistence, auth guards.
Memory is handled by LangGraph checkpointing (thread_id = session_id).
Traceability is handled by LangSmith (set env vars to enable).
"""
import json
import os
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.schemas import AgentEvent
from app.agents.supervisor_agent import run_supervisor_stream, run_supervisor_sync
from app.core.logger import get_logger
from app.schemas.chat import (
    ChatStreamRequest,
    MessageListResponse,
    MessageRead,
    SessionCreate,
    SessionListResponse,
    SessionRead,
)
from app.utils.chat_crud import (
    add_message,
    create_session,
    delete_session,
    get_messages,
    get_session,
    list_sessions,
    touch_session,
    update_session_title,
)
from app.utils.database import User, get_async_session
from app.utils.user import current_active_user

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])

# ---------------------------------------------------------------------------
# LangSmith run URL helper
# ---------------------------------------------------------------------------

def _build_langsmith_url(run_id: str) -> Optional[str]:
    """
    Build a direct link to a LangSmith run from its run_id.
    Only works when LANGSMITH_PROJECT (or LANGCHAIN_PROJECT) is set.
    """
    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
    if not project or not run_id:
        return None
    # LangSmith public URL format
    return f"https://smith.langchain.com/public/{run_id}/r"


# ---------------------------------------------------------------------------
# Small request/response helpers
# ---------------------------------------------------------------------------

class SimpleChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response_type: Optional[str]
    content: Optional[object]
    error: Optional[str] = None
    trace_url: Optional[str] = None


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: AgentEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _detect_agent(logs: List[str]) -> Optional[str]:
    """Best-effort guess at which tool dominated."""
    for log in logs:
        lower = log.lower()
        if "using tool: rag_tool" in lower:
            return "rag"
        if "using tool: sql_agent_tool" in lower:
            return "sql"
        if any(t in lower for t in (
            "using tool: search_tool",
            "using tool: extract_tool",
            "using tool: crawl_tool",
            "using tool: map_tool",
            "using tool: research_tool",
            "using tool: get_research_tool",
        )):
            return "web"
    return None


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------

@router.get("/sessions", response_model=SessionListResponse)
async def list_user_sessions(
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
):
    """List all chat sessions for the authenticated user (newest first)."""
    sessions = await list_sessions(db, str(user.id))
    session_reads = []
    for s in sessions:
        msgs = await get_messages(db, s.id, str(user.id))
        preview = msgs[-1].content[:100] if msgs else ""
        session_reads.append(
            SessionRead(
                id=s.id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
                preview=preview,
            )
        )
    return SessionListResponse(sessions=session_reads)


@router.post("/sessions", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def create_user_session(
    body: SessionCreate = SessionCreate(),
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Create a new chat session."""
    session = await create_session(db, str(user.id), body.title)
    return SessionRead(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        preview=None,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_session(
    session_id: str,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Delete a session and all its messages."""
    ok = await delete_session(db, session_id, str(user.id))
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Return all messages in a session (oldest first)."""
    session = await get_session(db, session_id, str(user.id))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = await get_messages(db, session_id, str(user.id))
    return MessageListResponse(
        messages=[MessageRead.model_validate(m) for m in msgs]
    )


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

async def _event_generator(
    request: SimpleChatRequest,
    user_id: str,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """Wrap the agent stream as SSE strings, persisting messages to DB."""
    session_id = request.session_id
    message = request.message
    tool_logs: List[str] = []

    try:
        # Persist user message
        if session_id:
            await add_message(db, session_id, "user", message)
            await touch_session(db, session_id)

        final_content: Optional[str] = None
        final_response_type: Optional[str] = None
        answer_done = False  # flag: stop yielding but keep draining

        # LangGraph checkpointer handles memory via thread_id = session_id
        async for event in run_supervisor_stream(message, thread_id=session_id):
            # After the answer, keep consuming run_supervisor_stream to its
            # natural end so we never throw GeneratorExit into it (which would
            # propagate into LangGraph and be recorded as a LangSmith error).
            if answer_done:
                continue

            yield _sse(event)

            if event.type == "log":
                tool_logs.append(str(event.content))

            elif event.type == "answer":
                content_str = (
                    json.dumps(event.content)
                    if isinstance(event.content, dict)
                    else str(event.content)
                )
                final_content = content_str
                final_response_type = event.response_type

                # Get LangSmith trace URL if tracing is enabled
                trace_url: Optional[str] = None
                try:
                    from langsmith import Client as LangSmithClient
                    ls = LangSmithClient()
                    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT", "default")
                    runs = list(ls.list_runs(
                        project_name=project,
                        filter=f'and(eq(metadata_key, "session_id"), eq(metadata_value, "{session_id}"))',
                        limit=1,
                        is_root=True,
                    ))
                    if runs:
                        trace_url = f"https://smith.langchain.com/public/{runs[0].id}/r"
                except Exception:
                    pass  # Tracing is optional — never break the chat

                # Persist assistant answer
                if session_id:
                    agent_used = _detect_agent(tool_logs)
                    await add_message(
                        db,
                        session_id,
                        "assistant",
                        content_str,
                        response_type=final_response_type,
                        agent_used=agent_used,
                        trace_url=trace_url,
                    )
                    # Auto-title: set from first user message if still "New Chat"
                    session = await get_session(db, session_id, user_id)
                    if session and session.title == "New Chat":
                        auto_title = message[:60] + ("..." if len(message) > 60 else "")
                        await update_session_title(db, session_id, auto_title)

                    # Emit a special metadata SSE event with the trace URL
                    if trace_url:
                        meta_event = AgentEvent(type="log", content=f"__trace_url__:{trace_url}")
                        yield _sse(meta_event)

                answer_done = True  # drain remaining events silently; no break


    except Exception as exc:
        logger.exception("Unhandled error in event generator")
        error_event = AgentEvent(type="error", content=str(exc))
        yield _sse(error_event)
    finally:
        yield _sse_done()


@router.post(
    "/chat/stream",
    summary="Chat (streaming SSE)",
    description=(
        "Stream the agent's reasoning steps and final answer as Server-Sent Events. "
        "Requires authentication. Each event is a JSON-serialised AgentEvent.\n\n"
        "Event types:\n"
        "- **log** — tool invocation notice (or __trace_url__:<url> for the LangSmith link)\n"
        "- **answer** — final answer (response_type + content)\n"
        "- **error** — something went wrong\n\n"
        "The stream ends with `data: [DONE]`."
    ),
)
async def chat_stream(
    request: SimpleChatRequest,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
) -> StreamingResponse:
    """Streaming SSE chat endpoint — recommended for the chat UI."""
    return StreamingResponse(
        _event_generator(request, str(user.id), db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat (blocking JSON)",
    description=(
        "Run the agent to completion and return a single JSON response. "
        "Simpler to consume but no streaming progress. "
        "Use `/chat/stream` for the chat UI."
    ),
)
async def chat(
    request: SimpleChatRequest,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
) -> ChatResponse:
    """Blocking chat endpoint — waits for the full agent run."""
    session_id = request.session_id
    message = request.message

    # Persist user message
    if session_id:
        await add_message(db, session_id, "user", message)

    try:
        # LangGraph checkpointer handles memory via thread_id
        event = await run_supervisor_sync(message, thread_id=session_id)
    except Exception as exc:
        logger.exception("Unhandled error in blocking chat endpoint")
        raise HTTPException(status_code=500, detail=str(exc))

    if event.type == "error":
        return ChatResponse(response_type=None, content=None, error=str(event.content))

    trace_url: Optional[str] = None
    if session_id and event.type == "answer":
        content_str = (
            json.dumps(event.content)
            if isinstance(event.content, dict)
            else str(event.content)
        )
        await add_message(
            db, session_id, "assistant", content_str,
            response_type=event.response_type,
            trace_url=trace_url,
        )
        session = await get_session(db, session_id, str(user.id))
        if session and session.title == "New Chat":
            await update_session_title(db, session_id, message[:60])

    return ChatResponse(
        response_type=event.response_type,
        content=event.content,
        trace_url=trace_url,
    )