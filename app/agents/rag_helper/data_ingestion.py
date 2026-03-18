from langchain.tools import tool
from .config import (
    UPLOAD_BATCH_SIZE,
    SKLEARN_REPO_PATH,
)
from .chunk_and_embed import (
    chunk_sklearn_repo,
    build_vectors_and_docs,
    batch_upsert,
    upload_to_mongo,
)
from .embedding_client import embedder
from app.core.logger import get_logger

log = get_logger(__name__)

@tool
def ingest_data():
    '''Parse, embed, and upload scikit-learn code/docs into Pinecone and MongoDB.'''
    log.info(" Starting data ingestion...")
    
    #Create chunks for embedding
    chunks = chunk_sklearn_repo(SKLEARN_REPO_PATH)
    log.info("Chunked %d chunks from scikit-learn repo", len(chunks))

    # create embedding
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    log.info("Embedded %d texts", len(embeddings))

    # upload to DB
    pinecone_vectors, mongo_docs = build_vectors_and_docs(chunks, embeddings)
    batch_upsert(vectors=pinecone_vectors, batch_size=UPLOAD_BATCH_SIZE)
    upload_to_mongo(mongo_docs=mongo_docs)
    log.info(" Data ingestion complete (%d vectors, %d docs)", len(pinecone_vectors), len(mongo_docs))
