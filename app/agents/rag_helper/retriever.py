from pymongo import MongoClient
import os 
from .config import MONGO_COLLECTION,MONGO_DB,INDEX_NAME
from .embedding_client import embedder
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.tools import tool
from app.core.logger import get_logger

load_dotenv()
log = get_logger(__name__)
MIN_SCORE_THRESHOLD = 0.3 

_mongo_client = None
_mongo_collection = None
_pinecone_client = None
_pinecone_index = None

def get_mongo_collection():
    global _mongo_client, _mongo_collection
    if _mongo_collection is None:
        log.info("Connecting to MongoDB...")
        _mongo_client = MongoClient(os.getenv("MONGODB_URL"))
        db = _mongo_client[MONGO_DB]
        _mongo_collection = db[MONGO_COLLECTION]
        log.info("✅ MongoDB connected (db=%s, collection=%s)", MONGO_DB, MONGO_COLLECTION)
    return _mongo_collection

def get_pinecone_index():
    global _pinecone_client, _pinecone_index
    if _pinecone_index is None:
        log.info("Connecting to Pinecone (index=%s)...", INDEX_NAME)
        _pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = _pinecone_client.Index(INDEX_NAME)
        log.info("✅ Pinecone connected")
    return _pinecone_index

def retrieve_context(
    query: str,
    top_k: int = 5,
):
    """Retrieve relevant scikit-learn code/docs chunks from the knowledge base."""
    log.info("Retrieving context (query_len=%d, top_k=%d)", len(query), top_k)
    # Embed query
    query_embedding = embedder.embed_query(query)
    log.debug("Query embedded (dim=%d)", len(query_embedding))

    # Query Pinecone
    index = get_pinecone_index()
    res = index.query(
        vector=query_embedding,
        top_k=top_k*5,
        include_metadata=True,
    )

    # Filter out test files and secondary rankings
    matches = []
    for m in res["matches"]:
        file_path = m["metadata"].get("file", "").lower()
        # Skip if it's a test file or in a tests directory
        if "test" in file_path or "tests" in file_path:
            continue
        
        if m["score"] >= MIN_SCORE_THRESHOLD:
            matches.append(m)
            if len(matches) >= top_k:
                break

    # Fetch from MongoDB
    ids = [m["id"] for m in matches]
    mongo_collection = get_mongo_collection()
    docs = mongo_collection.find({"_id": {"$in": ids}})
    doc_map = {d["_id"]: d for d in docs}

    # Preserve ranking
    ordered_docs = []
    for m in matches:
        doc = doc_map.get(m["id"])
        if doc:
            ordered_docs.append({
                "id": m["id"],
                "score": m["score"],
                "text": doc["text"],
                "file": doc.get("file"),
            })

    log.info("✅ Retrieved %d docs from %d Pinecone matches (score >= %.2f)",
             len(ordered_docs), len(res["matches"]), MIN_SCORE_THRESHOLD)
    return ordered_docs
