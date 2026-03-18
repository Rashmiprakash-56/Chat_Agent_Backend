import re
from pathlib import Path

###### SKLEARN CHUNK CONFIG ######
SKLEARN_VERSION = "1.8.0"
AUTHORITY_MAP = {
    "code": 1.0,
    "docs": 0.7,
    "example": 0.4
}
# SKIP_CLASS_PREFIXES = ("Base", "_")
# SKIP_FILE_PREFIXES = ("_",)
SKIP_FILE_PREFIXES = ()
SKIP_CLASS_PREFIXES = ()
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SKLEARN_REPO_PATH = ROOT_DIR / "scikit-learn"

RST_HEADER_PATTERN = re.compile(r"^(.+)\n([=~-]{3,})$", re.MULTILINE)

### PINECONE EMBEDDING CONFIG ###
INDEX_NAME = "sklearn-code-rag"
DIMENSION = 768
METRIC = "cosine"
BATCH_SIZE = 96
UPLOAD_BATCH_SIZE = 100

### MONGODB CONFIG ###
MONGO_DB = 'database_main'
MONGO_COLLECTION = 'rag_document'

### Base and Fallback model config ####
MODEL_REGISTRY = {
    "supervisor": {
        "primary": {"model": "gemini-2.5-flash", "provider": "google_genai"},
        "fallbacks": [
            {"model": "llama-3.3-70b-versatile", "provider": "groq"},
            {"model": "qwen/qwen3-32b", "provider": "groq"},
            {"model": "openai/gpt-oss-120b", "provider": "groq"},
            {"model": "moonshotai/kimi-k2-instruct", "provider": "groq"},
            {"model": "meta-llama/llama-4-maverick-17b-128e-instruct", "provider": "groq"},
            {"model": "gemma-3-27b-it", "provider": "google_genai"},
            {"model": "gemma-3-4b-it", "provider": "google_genai"},
            {"model": "gemini-2.5-flash-lite", "provider": "google_genai"},
            {"model": "groq/compound-mini", "provider": "groq"},
            {"model": "openai/gpt-oss-20b", "provider": "groq"},
            {"model": "gemma-3-12b-it", "provider": "google_genai"},
            {"model": "llama-3.1-8b-instant", "provider": "groq"},
            {"model": "gemma-3-12b-it", "provider": "google_genai"},
            {"model": "llama-3.1-8b-instant", "provider": "groq"},
            {"model": "groq/compound-mini", "provider": "groq"},
            {"model": "gemma-3-4b-it", "provider": "google_genai"},
            {"model": "openai/gpt-oss-20b", "provider": "groq"},
             {"model": "gemini-2.5-flash-lite", "provider": "google_genai"},
        ],
    },
    "sql": {
        "primary": {"model": "meta-llama/llama-4-scout-17b-16e-instruct", "provider": "groq"},
        "fallbacks": [
            {"model": "llama-3.3-70b-versatile", "provider": "groq"},
            {"model": "gemini-2.5-flash", "provider": "google_genai"},
            {"model": "qwen/qwen3-32b", "provider": "groq"},
            {"model": "groq/compound", "provider": "groq"},
            {"model": "openai/gpt-oss-120b", "provider": "groq"},
            {"model": "gemma-3-27b-it", "provider": "google_genai"},
        ],
    }
}

DEFAULT_TEMPERATURE = 0
