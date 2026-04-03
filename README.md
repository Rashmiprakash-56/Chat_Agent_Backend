---
title: Chat Agent Backend
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# 🤖 Chat Agent — Backend API

FastAPI backend for an AI chat agent powered by **LangGraph** with multi-agent orchestration, RAG, SQL querying, web search, and deep research capabilities.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Agent System](#agent-system)
- [Database Schema](#database-schema)
- [Deployment](#deployment)
- [Observability](#observability)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                        FastAPI Server                         │
│  (CORS · JWT Auth · Session Management · SSE Streaming)      │
├───────────────┬───────────────────────────────────────────────┤
│  Auth Router  │              Chat Router                      │
│  /auth/jwt/*  │  /agent/chat/stream (SSE)  /agent/chat (JSON)│
│  /auth/*      │  /agent/sessions/*                            │
│  /users/*     │                                               │
├───────────────┴───────────────────────────────────────────────┤
│                    Supervisor Agent (LangGraph)                │
│          ┌──────────┬──────────────┬──────────────┐           │
│          │ RAG Tool │ SQL Agent    │ Web Search   │           │
│          │(Pinecone)│ (PostgreSQL) │ (Tavily)     │           │
│          └──────────┴──────────────┴──────────────┘           │
├───────────────────────────────────────────────────────────────┤
│  Checkpointer (PostgreSQL)  │  LangSmith Tracing (Optional)  │
└───────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
backend/
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions → Hugging Face Spaces
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI entry point, lifespan, CORS, router mounting
│   ├── agents/
│   │   ├── base_model.py           # Multi-provider LLM factory with automatic fallbacks
│   │   ├── supervisor_agent.py     # LangGraph supervisor: orchestrates all tools
│   │   ├── sql_agent.py            # Natural language → SQL query execution
│   │   ├── models/
│   │   │   └── schemas.py          # AgentEvent, SearchResponse, CodeResponse schemas
│   │   ├── rag_helper/
│   │   │   ├── config.py           # Model registry, embedding config, Pinecone settings
│   │   │   ├── data_ingestion.py   # Document loader for RAG corpus
│   │   │   ├── chunk_and_embed.py  # Text splitting + vector embedding pipeline
│   │   │   ├── embedding_client.py # Embedding model client abstraction
│   │   │   └── retriever.py        # Pinecone-backed semantic retriever
│   │   └── tools/
│   │       ├── rag_tool.py         # RAG retrieval tool (scikit-learn docs)
│   │       └── search_tool.py      # Web search, crawl, extract, map, research tools
│   ├── core/
│   │   ├── config.py               # App settings (DB URLs, JWT, CORS)
│   │   └── logger.py               # Structured logging via structlog
│   ├── models/
│   │   ├── chat.py                 # SQLAlchemy models: ChatSession, ChatMessage
│   │   └── user.py                 # SQLAlchemy model: User (FastAPI Users)
│   ├── routers/
│   │   └── chat_router.py          # All /agent/* endpoints (sessions, chat, streaming)
│   ├── schemas/
│   │   ├── chat.py                 # Pydantic schemas for sessions & messages
│   │   └── user.py                 # Pydantic schemas for user CRUD
│   ├── services/
│   │   ├── preprocessing.py        # Data preprocessing utilities
│   │   └── train_model.py          # Model training utilities
│   └── utils/
│       ├── chat_crud.py            # Session & message CRUD operations (async)
│       ├── database.py             # Async SQLAlchemy engine & session factory
│       └── user.py                 # FastAPI Users config (auth backend, strategies)
├── Dockerfile                      # Production container (Python 3.13-slim)
├── requirements.txt                # Pinned Python dependencies
└── .env                            # Environment variables (not committed)
```

---

## Features

### 🧠 Multi-Agent Orchestration
- **Supervisor Agent** — A LangGraph-powered orchestrator that intelligently delegates user queries to specialised sub-agents instead of answering everything itself.
- **Dynamic Tool Selection** — The supervisor analyses each query and selects the minimal set of tools needed (RAG, SQL, web search, deep research).
- **Synthesised Output** — All tool results are synthesised into a single, structured final answer via the `submit_answer` tool.

### 📚 Retrieval-Augmented Generation (RAG)
- **Internal Knowledge Base** — Dedicated RAG pipeline tailored to scikit-learn documentation and source code.
- **Pinecone Vector Store** — Documents are ingested, chunked, embedded, and stored in Pinecone for semantic retrieval.
- **Configurable Embedding Models** — Abstracted embedding client supports multiple providers.

### 🗃️ Natural Language to SQL
- **Conversational Analytics** — Users ask questions in plain English; the SQL agent translates them to SQL queries.
- **Database Execution** — Queries execute against the connected PostgreSQL database with results returned as structured data.
- **Safety Guards** — DML detection middleware prevents destructive queries (INSERT, UPDATE, DELETE) without explicit approval.

### 🌐 Web Search & Deep Research
- **Tavily Integration** — Real-time web search for facts, current events, or anything outside the internal knowledge base.
- **Multi-Tool Research Suite** — `search_tool`, `extract_tool`, `crawl_tool`, `map_tool`, `research_tool`, and `get_research_tool` for deep, multi-step web research.

### 🔐 Authentication & Authorization
- **JWT-Based Auth** — Fully secured via FastAPI Users with Bearer token authentication.
- **User Isolation** — Each user can only access their own chat sessions and messages.
- **Full Auth Flow** — Registration, login, password reset, email verification, and user profile management.

### 💬 Real-Time Streaming (SSE)
- **Server-Sent Events** — The `/agent/chat/stream` endpoint streams reasoning steps and the final answer word-by-word.
- **Event Types** — `log` (tool invocation), `answer` (final response), `error` (failures).
- **Graceful Drain** — The stream drains LangGraph steps after the answer to prevent spurious LangSmith error traces.

### 🗂️ Session & Memory Management
- **Persistent Chat Sessions** — Sessions and messages stored in PostgreSQL via async SQLAlchemy.
- **Multi-Turn Memory** — LangGraph's checkpointer uses `session_id` as `thread_id` to maintain full conversation context across requests.
- **Auto-Titling** — Sessions are automatically titled from the user's first message.

### 📊 Observability & Tracing
- **LangSmith Integration** — Every LLM call, tool execution, and graph step is traced when enabled.
- **Trace URLs** — The API emits LangSmith trace URLs via SSE so the frontend can link directly to execution traces.
- **Structured Logging** — All application logs use `structlog` for consistent, parseable output.

### 🤖 Multi-Provider LLM Support
- **Automatic Fallbacks** — The LLM factory supports primary + fallback model chains (e.g., Gemini → Groq → OpenAI).
- **Model Registry** — Centralised configuration for all agent LLM providers and models.
- **Provider Agnostic** — Supports Google Gemini, Groq, Anthropic, OpenAI, and xAI via LangChain adapters.

---

## Tech Stack

| Category          | Technology                                                     |
| ----------------- | -------------------------------------------------------------- |
| **Framework**     | FastAPI 0.129, Uvicorn 0.38, Starlette 0.50                   |
| **AI/ML**         | LangChain ≥1.1, LangGraph ≥1.0, LangSmith ≥0.4               |
| **LLM Providers** | Google Gemini, Groq, Anthropic, OpenAI, xAI                   |
| **Vector Store**  | Pinecone ≥8.0                                                  |
| **Database**      | PostgreSQL (asyncpg), SQLAlchemy 2.0 (async)                   |
| **Auth**          | FastAPI Users 15.0, PyJWT, bcrypt, Argon2                      |
| **Search**        | Tavily Python 0.7                                               |
| **Streaming**     | SSE-Starlette 2.1                                               |
| **Observability** | structlog, OpenTelemetry, LangSmith                             |
| **Container**     | Docker (Python 3.13-slim)                                       |
| **CI/CD**         | GitHub Actions → Hugging Face Spaces                            |

---

## Getting Started

### Prerequisites

- **Python** 3.11+
- **PostgreSQL** instance (local or hosted, e.g., Supabase)
- **Pinecone** account (for RAG vector store)
- API keys for at least one LLM provider (Google Gemini, Groq, etc.)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd backend

# 2. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file (see Environment Variables below)
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux

# 5. Run the development server
uvicorn app.main:app --reload --port 7860
```

The API will be available at `http://localhost:7860`. Interactive docs at `http://localhost:7860/docs`.

---

## Environment Variables

Create a `.env` file in the backend root with the following keys:

```env
# ── Database ─────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname
LANGGRAPH_DB_URL=postgresql://user:password@host:5432/dbname

# ── JWT / Auth ───────────────────────────────────────────────
SECRET_KEY=your-secret-key-here

# ── CORS ─────────────────────────────────────────────────────
CORS_ORIGINS=http://localhost:8888,https://your-frontend.com

# ── LLM Providers (at least one required) ────────────────────
GOOGLE_API_KEY=your-google-api-key
GROQ_API_KEY=your-groq-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key       # optional
OPENAI_API_KEY=your-openai-api-key             # optional
XAI_API_KEY=your-xai-api-key                   # optional

# ── Pinecone (RAG) ──────────────────────────────────────────
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name

# ── Tavily (Web Search) ─────────────────────────────────────
TAVILY_API_KEY=your-tavily-api-key

# ── MongoDB (if used) ───────────────────────────────────────
MONGODB_URI=mongodb+srv://user:password@cluster/dbname

# ── LangSmith (optional observability) ──────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=Chat_Agent
```

---

## API Reference

### Health & Info

| Method | Endpoint    | Description          |
| ------ | ----------- | -------------------- |
| GET    | `/`         | API info & version   |
| GET    | `/health`   | Health check         |

### Authentication (`/auth`)

| Method | Endpoint              | Description                |
| ------ | --------------------- | -------------------------- |
| POST   | `/auth/jwt/login`     | Login → JWT token          |
| POST   | `/auth/jwt/logout`    | Logout (invalidate token)  |
| POST   | `/auth/register`      | Register new user          |
| POST   | `/auth/forgot-password` | Request password reset   |
| POST   | `/auth/reset-password`  | Reset password with token |
| POST   | `/auth/verify`        | Verify email address       |

### Users (`/users`)

| Method | Endpoint      | Description            |
| ------ | ------------- | ---------------------- |
| GET    | `/users/me`   | Get current user       |
| PATCH  | `/users/me`   | Update current user    |

### Chat Sessions (`/agent`)

| Method | Endpoint                                 | Description                          |
| ------ | ---------------------------------------- | ------------------------------------ |
| GET    | `/agent/sessions`                        | List all sessions (newest first)     |
| POST   | `/agent/sessions`                        | Create a new session                 |
| DELETE | `/agent/sessions/{session_id}`           | Delete a session and its messages    |
| GET    | `/agent/sessions/{session_id}/messages`  | Get all messages in a session        |

### Chat (`/agent`)

| Method | Endpoint              | Description                                  |
| ------ | --------------------- | -------------------------------------------- |
| POST   | `/agent/chat/stream`  | **SSE streaming** — real-time agent response |
| POST   | `/agent/chat`         | **Blocking JSON** — full response at once    |

#### SSE Stream Event Format

```json
{"type": "log",    "content": "Using tool: rag_tool"}
{"type": "log",    "content": "Using tool: search_tool"}
{"type": "answer", "response_type": "text", "content": {"answer": "...", "source_urls": [...]}}
{"type": "log",    "content": "__trace_url__:https://smith.langchain.com/public/..."}
```

The stream ends with: `data: [DONE]`

---

## Agent System

### Supervisor Agent

The central orchestrator built with LangGraph's `create_agent`. It receives user queries and delegates to the appropriate tool(s):

```
User Query → Supervisor Agent → [Tool Selection] → [Tool Execution] → [Synthesis] → submit_answer
```

### Available Tools

| Tool               | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `rag_tool`         | Query the internal scikit-learn knowledge base (Pinecone)  |
| `sql_agent_tool`   | Translate natural language to SQL and execute queries       |
| `search_tool`      | General web search via Tavily                              |
| `extract_tool`     | Extract content from specific web pages                    |
| `crawl_tool`       | Crawl web pages for information                            |
| `map_tool`         | Map website structure and content                          |
| `research_tool`    | Conduct multi-step deep research                           |
| `get_research_tool`| Retrieve results from ongoing research                     |
| `submit_answer`    | Submit the final structured response (required to end run) |

### Response Types

- **`text`** → `SearchResponse { answer: string, source_urls: string[] }`
- **`code`** → `CodeResponse { language: string, code: string, explanation: string }`

### Multi-Turn Memory

The LangGraph checkpointer (backed by PostgreSQL) maps each `session_id` to a `thread_id`. This means:
- The agent automatically loads prior conversation history for follow-up queries.
- No manual context management is needed — LangGraph handles it transparently.

### LLM Fallback Chain

Each agent role has a primary model and a list of fallback models. If the primary provider fails (rate limit, downtime), the system automatically retries with the next provider in the chain.

---

## Database Schema

### PostgreSQL Tables

| Table            | Purpose                                     |
| ---------------- | ------------------------------------------- |
| `user`           | User accounts (FastAPI Users)               |
| `chat_session`   | Chat sessions (id, user_id, title, timestamps) |
| `chat_message`   | Messages (id, session_id, role, content, response_type, agent_used, trace_url) |
| `checkpoint_*`   | LangGraph checkpointer tables (auto-managed) |

---

## Deployment

### Docker

```bash
docker build -t chat-agent-backend .
docker run -p 7860:7860 --env-file .env chat-agent-backend
```

### Hugging Face Spaces (CI/CD)

The project auto-deploys to Hugging Face Spaces on every push to `main` via GitHub Actions:

- **Workflow**: `.github/workflows/deploy.yml`
- **Space**: `Rashmiprakash78/Chat_Agent`
- **Trigger**: Push to `main` branch
- **Requires**: `HF_TOKEN` secret configured in GitHub repository settings

---

## Observability

### LangSmith Tracing

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set:
- Every LLM call, prompt, and tool execution is logged to LangSmith.
- Trace URLs are emitted in the SSE stream for frontend deep-linking.
- Access traces at: `https://smith.langchain.com`

### Structured Logging

All application logs use `structlog` for consistent, JSON-parseable output with context binding (request IDs, user IDs, etc.).

### OpenTelemetry

OpenTelemetry SDK is included for distributed tracing and metrics export via OTLP.

---

## License

This project is for educational and personal use.
