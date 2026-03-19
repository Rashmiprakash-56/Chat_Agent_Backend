---
title: Chat Agent Backend
emoji: 🤖
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
---

<div align="center">

# 🤖 Chat Agent — Backend

**An intelligent, multi-agent conversational AI backend powered by LangGraph, FastAPI, and RAG.**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Deploy](https://img.shields.io/badge/🤗-HF%20Spaces-yellow)](https://huggingface.co/spaces/Rashmiprakash78/Chat_Agent)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)

---

## Overview

Chat Agent Backend is a production-ready AI agent API that orchestrates multiple specialized sub-agents through a **supervisor architecture**. It supports:

- 🔍 **RAG (Retrieval-Augmented Generation)** — Document ingestion, chunking, embedding, and semantic retrieval via Pinecone
- 🗃️ **SQL Agent** — Natural-language-to-SQL querying against PostgreSQL databases
- 🌐 **Web Search** — Real-time web search powered by Tavily
- 💬 **Multi-turn Conversations** — Persistent chat history with checkpointed state via LangGraph
- 🔐 **Authentication** — Full user auth (register, login, JWT, password reset) via FastAPI Users
- 📊 **Observability** — Structured logging with `structlog` and tracing via OpenTelemetry + LangSmith

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   FastAPI App                    │
│                  (app/main.py)                   │
├──────────┬──────────────────────┬────────────────┤
│  Auth    │    Chat Router       │   Health       │
│ /auth/*  │    /agent/*          │   /health      │
└──────────┴──────────┬───────────┴────────────────┘
                      │
            ┌─────────▼─────────┐
            │  Supervisor Agent │
            │  (LangGraph)      │
            └───┬───────┬───────┘
                │       │
     ┌──────────┤       ├──────────┐
     ▼          ▼       ▼          ▼
 ┌───────┐ ┌───────┐ ┌──────┐ ┌────────┐
 │  RAG  │ │  SQL  │ │ Web  │ │ Base   │
 │ Tool  │ │ Agent │ │Search│ │ Model  │
 └───┬───┘ └───┬───┘ └──┬───┘ └────────┘
     │         │        │
     ▼         ▼        ▼
 Pinecone  PostgreSQL  Tavily
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Framework** | FastAPI, Uvicorn, Pydantic |
| **AI / Agents** | LangChain, LangGraph, LangSmith |
| **LLM Providers** | Google Gemini, Groq, Anthropic, OpenAI, xAI |
| **Vector Store** | Pinecone |
| **Databases** | PostgreSQL (Supabase), MongoDB, SQLite (checkpoints) |
| **Auth** | FastAPI Users, JWT, bcrypt, Argon2 |
| **Observability** | structlog, OpenTelemetry, LangSmith |
| **Deployment** | Docker, GitHub Actions → Hugging Face Spaces |

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── agents/
│   │   ├── supervisor_agent.py  # LangGraph supervisor orchestrator
│   │   ├── sql_agent.py         # Natural language → SQL agent
│   │   ├── base_model.py        # Base LLM model configuration
│   │   ├── models/
│   │   │   └── schemas.py       # Agent-level Pydantic schemas
│   │   ├── rag_helper/
│   │   │   ├── config.py        # RAG pipeline configuration
│   │   │   ├── data_ingestion.py    # Document loading
│   │   │   ├── chunk_and_embed.py   # Text chunking & embedding
│   │   │   ├── embedding_client.py  # Embedding API client
│   │   │   └── retriever.py         # Pinecone vector retrieval
│   │   └── tools/
│   │       ├── rag_tool.py      # RAG LangChain tool
│   │       └── search_tool.py   # Web search tool (Tavily)
│   ├── core/
│   │   ├── config.py            # App settings (Pydantic Settings)
│   │   └── logger.py            # Structured logging setup
│   ├── models/
│   │   ├── chat.py              # Chat SQLAlchemy models
│   │   └── user.py              # User SQLAlchemy models
│   ├── routers/
│   │   └── chat_router.py       # Chat / agent API routes
│   ├── schemas/
│   │   ├── chat.py              # Chat Pydantic schemas
│   │   └── user.py              # User Pydantic schemas
│   ├── services/
│   │   ├── preprocessing.py     # Data preprocessing utilities
│   │   └── train_model.py       # ML model training service
│   └── utils/
│       ├── database.py          # Async SQLAlchemy engine & session
│       ├── chat_crud.py         # Chat CRUD operations
│       └── user.py              # FastAPI Users backend config
├── .github/
│   └── workflows/
│       └── deploy.yml           # CI/CD → Hugging Face Spaces
├── Dockerfile                   # Production Docker image
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not committed)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- **Python 3.13+**
- **PostgreSQL** database (or Supabase)
- **Pinecone** account (for vector storage)
- API keys for at least one LLM provider (Gemini, Groq, etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rashmigit/Chat_Agent_Backend.git
   cd Chat_Agent_Backend
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

5. **Run the development server**
   ```bash
   uvicorn app.main:app --reload --port 7860
   ```

6. **Open the API docs**
   
   Navigate to [http://localhost:7860/docs](http://localhost:7860/docs) for the interactive Swagger UI.

---

## Environment Variables

Create a `.env` file in the project root with the following variables:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google Gemini API key |
| `XAI_API_KEY` | xAI (Grok) API key |
| `GROQ_API_KEY` | Groq API key |
| `PINECONE_API_KEY` | Pinecone vector database key |
| `EMBEDDING_URL` | URL for the embedding model endpoint |
| `MONGODB_URL` | MongoDB connection string |
| `TAVILY_API_KEY` | Tavily web search API key |
| `DATABASE_URL` | PostgreSQL connection string (async, `postgresql+psycopg://...`) |
| `LANGGRAPH_DB_URL` | PostgreSQL URL for LangGraph checkpointer |
| `SQL_DATABASE_URL` | PostgreSQL URL for the SQL agent |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing (`true` / `false`) |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_PROJECT` | LangSmith project name |
| `SECRET_KEY` | JWT signing secret |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS origins |

---

## API Endpoints

### Health & Root

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info & status |
| `GET` | `/health` | Health check |

### Authentication (`/auth`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/jwt/login` | Login (returns JWT) |
| `POST` | `/auth/jwt/logout` | Logout |
| `POST` | `/auth/register` | Register a new user |
| `POST` | `/auth/forgot-password` | Request password reset |
| `POST` | `/auth/reset-password` | Reset password |
| `POST` | `/auth/verify` | Verify user email |

### Users (`/users`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/users/me` | Get current user profile |
| `PATCH` | `/users/me` | Update current user |

### Agent (`/agent`)

| Method | Endpoint | Description |
|---|---|---|
| Defined in `chat_router.py` | `/agent/*` | Chat, conversation history, and agent interaction endpoints |

> 📖 For full request/response schemas, visit the **Swagger UI** at `/docs` when the server is running.

---

## Deployment

### Docker

```bash
# Build the image
docker build -t chat-agent-backend .

# Run the container
docker run -p 7860:7860 --env-file .env chat-agent-backend
```

### Hugging Face Spaces (CI/CD)

The project includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically deploys to [Hugging Face Spaces](https://huggingface.co/spaces/Rashmiprakash78/Chat_Agent) on every push to `main`.

**Setup:**
1. Add `HF_TOKEN` as a GitHub repository secret
2. Push to the `main` branch — deployment is automatic

---

<div align="center">

Made with ❤️ using FastAPI + LangGraph

</div>
