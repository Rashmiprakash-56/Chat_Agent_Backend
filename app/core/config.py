from typing import List
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings():
    # Database      
    DATABASE_URL: str = os.getenv('DATABASE_URL')
    LANGGRAPH_DB_URL: str = os.getenv('LANGGRAPH_DB_URL')
    
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = "HS256" 
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # CORS — set CORS_ORIGINS env var as comma-separated URLs for production
    # Defaults to ["*"] for local development
    _cors_env = os.getenv("CORS_ORIGINS", "")
    CORS_ORIGINS: List[str] = (
        [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
        if _cors_env
        else ["*"]
    )

    
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    class Config:
        env_file = ".env"
    
settings = Settings()