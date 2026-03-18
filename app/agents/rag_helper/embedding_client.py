import requests
import os
import time
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from app.core.logger import get_logger

load_dotenv()
log = get_logger(__name__)

class HFSpaceEmbeddingClient:
    def __init__(
        self,
        endpoint_url: str,
        dimension: int,
        batch_size: int = 32,
        timeout: int = 60,
        retry: int = 3,
        sleep_between_retries: float = 1.0,
    ):
        self.endpoint_url = endpoint_url
        self.dimension = dimension
        self.batch_size = batch_size
        self.timeout = timeout
        self.retry = retry
        self.sleep = sleep_between_retries

    # ---------- internal ----------
    def _post(self, texts: List[str]) -> List[List[float]]:
        payload = {"texts": texts}

        for attempt in range(self.retry):
            try:
                r = requests.post(
                    self.endpoint_url,
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()

                embeddings = data["embeddings"]
                if len(embeddings[0]) != self.dimension:
                    raise ValueError("Embedding dimension mismatch")

                return embeddings

            except Exception as exc:
                log.warning("Embedding request failed (attempt %d/%d): %s", attempt + 1, self.retry, exc)
                if attempt == self.retry - 1:
                    raise
                time.sleep(self.sleep)

    # ---------- public ----------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed many documents (used for chunk ingestion).
        Shows a progress bar.
        """
        log.info("Embedding %d documents (batch_size=%d)", len(texts), self.batch_size)
        embeddings = []
        total = len(texts)

        with tqdm(total=total, desc="Embedding chunks", unit="chunk") as pbar:
            for i in range(0, total, self.batch_size):
                batch = texts[i : i + self.batch_size]
                embs = self._post(batch)

                embeddings.extend(embs)
                pbar.update(len(batch))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query (used at retrieval time).
        No progress bar (fast path).
        """
        return self._post([text])[0]


embedder = HFSpaceEmbeddingClient(
    endpoint_url= os.getenv("EMBEDDING_URL"),
    dimension=768,
    batch_size=96,
)
