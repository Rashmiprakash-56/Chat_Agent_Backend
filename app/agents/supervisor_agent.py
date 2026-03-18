"""
agents/supervisor_agent.py

Supervisor agent that orchestrates all tools and streams results
to the FastAPI layer via an async generator.

Memory is handled by LangGraph's AsyncSqliteSaver checkpointer —
each session_id maps to a unique thread_id so the graph
automatically loads and saves conversation state across turns.

Traceability is handled by LangSmith (set LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY in your .env to enable).
"""

import asyncio
import traceback
from typing import AsyncGenerator, Optional, Union

from app.core.logger import get_logger

log = get_logger(__name__)

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.core.config import settings

from .base_model import get_llm_with_fallbacks
from .models.schemas import AgentEvent, CodeResponse, SearchResponse
from .sql_agent import sql_agent_tool
from .tools.rag_tool import rag_tool
from .tools.search_tool import (
    crawl_tool,
    extract_tool,
    get_research_tool,
    map_tool,
    research_tool,
    search_tool,
)

# ---------------------------------------------------------------------------
# Checkpointer (LangGraph persistent memory)
# ---------------------------------------------------------------------------

# checkpointer.py

from typing import Optional
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

_checkpointer: Optional[AsyncPostgresSaver] = None
_checkpointer_cm = None

async def init_checkpointer() -> None:
    global _checkpointer, _checkpointer_cm

    if _checkpointer is None:
        log.info("Initializing Postgres checkpointer...")
        _checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.LANGGRAPH_DB_URL)
        _checkpointer = await _checkpointer_cm.__aenter__()

        await _checkpointer.setup()
        log.info("✅ Checkpointer initialized successfully")


async def close_checkpointer() -> None:
    global _checkpointer_cm

    if _checkpointer_cm is not None:
        log.info("Closing Postgres checkpointer...")
        await _checkpointer_cm.__aexit__(None, None, None)
        log.info("✅ Checkpointer closed")


def get_checkpointer() -> AsyncPostgresSaver:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialised.")
    return _checkpointer
# Single shared checkpointer — opened once at startup via init_checkpointer()
# _checkpointer: Optional[AsyncSqliteSaver] = None


# async def init_checkpointer() -> None:
#     """
#     Open the SQLite checkpointer.  Call once from the FastAPI lifespan event.
#     Using the same DB file as the rest of the app keeps the footprint small.
#     """
#     global _checkpointer
#     if _checkpointer is None:
#         _checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")


# def get_checkpointer() -> AsyncSqliteSaver:
#     if _checkpointer is None:
#         raise RuntimeError(
#             "Checkpointer not initialised — call await init_checkpointer() in lifespan."
#         )
#     return _checkpointer


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def submit_answer(
    response_type: str,
    content: Union[SearchResponse, CodeResponse],
) -> str:
    """
    Submit the final, structured answer to the user.
    This MUST be the last tool called to end the agent loop.

    Args:
        response_type: "text" for prose answers, "code" for code outputs.
        content:       SearchResponse (text) or CodeResponse (code).
    """
    return "__ANSWER_SUBMITTED__"


SUPERVISOR_TOOLS = [
    rag_tool,
    sql_agent_tool,
    search_tool,
    extract_tool,
    crawl_tool,
    map_tool,
    research_tool,
    get_research_tool,
    submit_answer,
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Supervisor Agent — an expert orchestrator of specialised tools.
Your job is to answer the user's request accurately and efficiently by delegating to the right tool(s).

## Conversation Memory
You receive the full prior conversation as part of your message history.
Use it to answer follow-up questions in context — e.g. "what about X?" or "give me a code example"
refers to the topic already discussed. Do NOT re-introduce yourself or repeat prior answers verbatim.

## Tools
| Tool | When to use |
|------|-------------|
| rag_tool | Internal knowledge base (scikit-learn docs/code). Use for library-specific questions. |
| sql_agent_tool | Structured database queries — counting, filtering, aggregation. |
| search_tool | General web search for facts, current events, or anything not in internal docs. |
| extract_tool / crawl_tool / map_tool / research_tool / get_research_tool | Deeper web research when a single search isn't enough. |

## Rules
1. Think before acting — choose the minimal set of tools needed.
2. Never repeat a tool call with the same arguments.
3. Synthesise all tool outputs into one clear, helpful final answer.
4. You MUST end every task by calling `submit_answer`:
   - User asked a question  → response_type="text",  content=SearchResponse(...)
   - User asked for code    → response_type="code",  content=CodeResponse(...)
5. Never leave a task unfinished without calling `submit_answer`.
"""

# ---------------------------------------------------------------------------
# Agent singleton
# ---------------------------------------------------------------------------

_supervisor_agent_instance = None


def get_supervisor_agent():
    global _supervisor_agent_instance
    if _supervisor_agent_instance is None:
        log.info("Creating supervisor agent (first request)...")
        llm = get_llm_with_fallbacks("supervisor")
        _supervisor_agent_instance = create_agent(
            model=llm,
            tools=SUPERVISOR_TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=get_checkpointer(),
        )
        log.info("✅ Supervisor agent created with %d tools", len(SUPERVISOR_TOOLS))
    return _supervisor_agent_instance


# Maximum graph steps before we abort (prevents runaway agents in production)
_MAX_STEPS = 50


# ---------------------------------------------------------------------------
# Public interface: async streaming generator
# ---------------------------------------------------------------------------

async def run_supervisor_stream(
    query: str,
    thread_id: Optional[str] = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Run the supervisor agent and yield AgentEvent objects.

    thread_id: the chat session_id.  The LangGraph checkpointer uses this to
               automatically load and persist conversation history so the
               agent has full multi-turn memory across requests.

    Event types
    -----------
    log    — intermediate thought or tool invocation notice
    answer — final structured answer (response_type + content)
    error  — unrecoverable failure with detail message

    Implementation note — why we drain instead of break
    ----------------------------------------------------
    LangGraph's astream wraps its entire body in ``except BaseException as e``
    and calls ``run_manager.on_chain_error(e)`` before re-raising.  That
    callback is the LangSmith hook that records trace errors.  ``GeneratorExit``
    is a BaseException, so ANY early exit from astream (break, return, aclose)
    is reported to LangSmith as an error.

    The only way to prevent the spurious error trace is to let astream run
    to its natural end.  After we receive the answer event we therefore stop
    *yielding* but keep *consuming* the remaining steps until the graph finishes.
    """
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    input_payload = {"messages": [{"role": "user", "content": query}]}
    steps = 0
    log.info(" Supervisor stream started (thread=%s, query_len=%d)", thread_id, len(query))
    answer_received = False

    try:
        agent = get_supervisor_agent()
        async for step in agent.astream(input_payload, config=config, stream_mode="updates"):
            steps += 1

            # Hard safety cap — yield the error once then drain silently.
            if steps > _MAX_STEPS:
                if not answer_received:
                    log.warning("⚠️ Agent exceeded max step limit (%d) without answer", _MAX_STEPS)
                    yield AgentEvent(
                        type="error",
                        content="Agent exceeded maximum step limit without producing an answer.",
                    )
                    answer_received = True  # treat as done; keep draining
                continue  # drain remaining steps without yielding

            # Once we have an answer, keep consuming so astream finishes
            # naturally — never break early (see Implementation note above).
            if answer_received:
                continue

            for _node, node_update in step.items():
                if not isinstance(node_update, dict):
                    continue
                for msg in node_update.get("messages", []):
                    async for event in _parse_message(msg):
                        yield event
                        if event.type == "answer":
                            answer_received = True
                            log.info("✅ Answer received at step %d", steps)

    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected — nothing to surface, just stop.
        log.info("Client disconnected (thread=%s)", thread_id)
    except Exception as exc:
        log.exception("❌ Supervisor stream error (thread=%s): %s", thread_id, exc)
        yield AgentEvent(
            type="error",
            content=str(exc),
            detail=traceback.format_exc(),
        )


async def run_supervisor_sync(
    query: str,
    thread_id: Optional[str] = None,
) -> AgentEvent:
    """
    Non-streaming variant — collects the full agent run and returns only
    the final AgentEvent (answer or error).  Useful for simple REST endpoints.
    """
    last_event: AgentEvent | None = None
    async for event in run_supervisor_stream(query, thread_id=thread_id):
        last_event = event
        if event.type == "answer":
            return event

    return last_event or AgentEvent(type="error", content="Agent produced no output.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _parse_message(message) -> AsyncGenerator[AgentEvent, None]:
    """Translate a single LangChain message object into AgentEvent(s)."""
    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls:
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args", {})

            if name == "submit_answer":
                yield AgentEvent(
                    type="answer",
                    response_type=args.get("response_type"),
                    content=args.get("content"),
                )
                return  # nothing after the final answer

            # Any other tool — surface it as a log so the UI can show progress
            yield AgentEvent(type="log", content=f"Using tool: {name}")
        return  # don't fall through when tool_calls are present

    # Plain text (reasoning / intermediate thoughts)
    text = getattr(message, "content", "")
    if isinstance(text, str) and text.strip():
        yield AgentEvent(type="log", content=text)
