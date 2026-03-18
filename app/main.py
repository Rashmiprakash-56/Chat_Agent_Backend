import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from app.core.config import settings
from app.utils.database import create_db_and_table,get_async_session,User
from app.utils.user import auth_backend,current_active_user , fastapi_users
from app.schemas.user import UserCreate,UserRead,UserUpdate
from app.core.logger import get_logger
from app.routers.chat_router import router as chat_router  
from dotenv import load_dotenv
import os

load_dotenv() 

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app : FastAPI):
    await create_db_and_table()
    from app.agents.supervisor_agent import init_checkpointer
    await init_checkpointer()
    log.info("🚀 Agent starting up")
    yield
    log.info("🛑 Agent  shutting down")
    from app.agents.supervisor_agent import close_checkpointer
    await close_checkpointer()


app = FastAPI(
    title="AI AGENT API",
    description="LangChain-powered agent with RAG, SQL, and web search tools.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"])
app.include_router(fastapi_users.get_register_router(UserRead,UserCreate), prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_users_router(UserRead,UserUpdate), prefix="/users", tags=["users"])
app.include_router(chat_router, prefix="/agent",tags=["agent"])


@app.get("/")
async def root():
    return {
        "message": "AI Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
    log.info("Server has started on port %s", port)