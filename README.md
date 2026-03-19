---
title: Chat Agent Backend
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Chat Agent — Backend API

FastAPI backend for an AI chat agent powered by LangGraph with RAG, SQL querying, and web search capabilities.

## Features

- **Multi-Agent Architecture** — Supervisor agent orchestrating RAG, SQL, and web search sub-agents
- **RAG Pipeline** — Document ingestion, chunking, and semantic retrieval via Pinecone
- **SQL Agent** — Natural language to SQL querying
- **Web Search** — Real-time search via Tavily
- **Authentication** — JWT-based auth with FastAPI Users
- **Observability** — Structured logging and LangSmith tracing

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

## Environment Variables

Requires a `.env` file with API keys for: Google Gemini, Groq, Pinecone, Tavily, MongoDB, PostgreSQL (Supabase), and LangSmith.

## Deployment

Automatically deployed to Hugging Face Spaces via GitHub Actions on push to `main`.
