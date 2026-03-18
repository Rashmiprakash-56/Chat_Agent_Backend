from langchain.tools import tool
from ..rag_helper.retriever import retrieve_context
from app.core.logger import get_logger

log = get_logger(__name__)

@tool
def rag_tool(query: str,top_k : int = 5) -> str:
    """
    Search the knowledge base (scikit-learn documentation and code) for relevant context.
    Use this tool when the user asks about specific library functionality, classes, or seeks code examples from the docs.
    """
    log.info("rag_tool invoked (query_len=%d, top_k=%d): %s", len(query), top_k, query[:100])
    docs = retrieve_context(query,top_k=top_k)
    log.info("✅ rag_tool retrieved %d docs", len(docs))
    
    formatted_docs = []
    for doc in docs:
        source = doc.get("file", "unknown")
        content = doc.get("text", "").strip()
        formatted_docs.append(f"Source: {source}\nContent:\n{content}\n")
    
    return "\n---\n".join(formatted_docs)
