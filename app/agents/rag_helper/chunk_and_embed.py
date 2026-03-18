import ast
import uuid
from pinecone import Pinecone, ServerlessSpec
import os
from pymongo import MongoClient
from pathlib import Path
from .config import (
    SKIP_CLASS_PREFIXES,SKIP_FILE_PREFIXES,AUTHORITY_MAP,SKLEARN_VERSION,RST_HEADER_PATTERN,
    INDEX_NAME,MONGO_DB,MONGO_COLLECTION,UPLOAD_BATCH_SIZE
    )

def extract_python_chunks(file_path, source_type):
    chunks = []
    source = file_path.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith(SKIP_CLASS_PREFIXES):
                continue

            code = ast.get_source_segment(source, node)
            if not code:
                continue

            chunks.append({
                "text": code,
                "name": node.name,
                "chunk_type": "class",
                "source": source_type,
                "authority": AUTHORITY_MAP[source_type],
                "visibility": "public",
                "file": str(file_path),
                "module": file_path.parent.name,
                "sklearn_version": SKLEARN_VERSION,
            })

        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue

            code = ast.get_source_segment(source, node)
            if not code:
                continue

            chunks.append({
                "text": code,
                "name": node.name,
                "chunk_type": "function",
                "source": source_type,
                "authority": AUTHORITY_MAP[source_type],
                "visibility": "public",
                "file": str(file_path),
                "module": file_path.parent.name,
                "sklearn_version": SKLEARN_VERSION,
            })

    return chunks

def extract_rst_chunks(file_path):
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    chunks = []

    matches = list(RST_HEADER_PATTERN.finditer(text))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        title = match.group(1).strip()

        chunks.append({
            "text": section_text,
            "name": title,
            "chunk_type": "doc_section",
            "source": "docs",
            "authority": AUTHORITY_MAP["docs"],
            "visibility": "public",
            "file": str(file_path),
            "sklearn_version": SKLEARN_VERSION,
        })

    return chunks

def extract_example_chunks(file_path):
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    return [{
        "text": text,
        "name": file_path.stem,
        "chunk_type": "example",
        "source": "example",
        "authority": AUTHORITY_MAP["example"],
        "visibility": "illustrative",
        "file": str(file_path),
        "sklearn_version": SKLEARN_VERSION,
    }]

def chunk_sklearn_repo(repo_path):
    repo_path = Path(repo_path)
    all_chunks = []

    for file in repo_path.rglob("*"):
        if not file.is_file():
            continue

        if file.name.startswith(SKIP_FILE_PREFIXES):
            continue

        # ---------- CODE ----------
        if "sklearn" in file.parts and file.suffix == ".py":
            all_chunks.extend(
                extract_python_chunks(file, source_type="code")
            )

        # ---------- DOCS ----------
        elif "doc" in file.parts and file.suffix == ".rst":
            all_chunks.extend(
                extract_rst_chunks(file)
            )

        # ---------- EXAMPLES ----------
        elif "examples" in file.parts and file.suffix in {".py", ".rst"}:
            continue
            # all_chunks.extend(
            #     extract_example_chunks(file)
            # )

    return all_chunks

def build_vectors_and_docs(chunks, embeddings):
    pinecone_vectors = []
    mongo_docs = []

    for chunk, emb in zip(chunks, embeddings):
        doc_id = str(uuid.uuid4())

        # ---- Pinecone (keep metadata SMALL) ----
        pinecone_vectors.append((
            doc_id,        # same ID used in Mongo
            emb,           # 768-dim embedding
            {
                "chunk_type": chunk["chunk_type"],
                "source": chunk["source"],
                "visibility": chunk["visibility"],
                "authority": chunk["authority"],
                "file": chunk["file"],
                "sklearn_version": chunk["sklearn_version"],
            }
        ))

        # ---- MongoDB (store heavy text + rich info) ----
        mongo_docs.append({
            "_id": doc_id,
            "text": chunk["text"],       
            "name": chunk["name"],
            "chunk_type": chunk["chunk_type"],
            "source": chunk["source"],
            "authority": chunk["authority"],
            "visibility": chunk["visibility"],
            "file": chunk["file"],
            "module": chunk.get("module", ""),
            "sklearn_version": chunk["sklearn_version"],
        })

    return pinecone_vectors, mongo_docs

def batch_upsert(vectors, batch_size=UPLOAD_BATCH_SIZE):
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = pc.Index(INDEX_NAME)
    DIMENSION = DIMENSION
    METRIC = METRIC

    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

def upload_to_mongo(mongo_docs):    

    # Insert into MongoDB
    client = MongoClient(os.getenv("MONGODB_URL"))
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    collection.insert_many(mongo_docs)


