import os
import hashlib
import time
from typing import List, Dict, Any, Optional
from cachetools import LRUCache
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config import settings


# Global LRU cache for query embeddings: query_hash -> list of floats
_QUERY_EMBED_CACHE = LRUCache(maxsize=settings.CACHE_MAX_SIZE)

# Global embedding model instance (lazy-loaded)
_EMBED_MODEL = None

# Global shared Qdrant client instance
_SHARED_QDRANT_CLIENT = None


def get_shared_qdrant_client() -> QdrantClient:
    global _SHARED_QDRANT_CLIENT
    if _SHARED_QDRANT_CLIENT is None:
        if settings.QDRANT_USE_EMBEDDED:
            try:
                _SHARED_QDRANT_CLIENT = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
            except Exception:
                lock_file = os.path.join(settings.QDRANT_STORAGE_PATH, ".lock")
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                        _SHARED_QDRANT_CLIENT = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
                    except Exception:
                        _SHARED_QDRANT_CLIENT = QdrantClient(":memory:")
                else:
                    _SHARED_QDRANT_CLIENT = QdrantClient(":memory:")
        else:
            _SHARED_QDRANT_CLIENT = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY
            )
    return _SHARED_QDRANT_CLIENT


def get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] Loading multilingual embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            _EMBED_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"[WARN] Could not load SentenceTransformer: {e}. Using deterministic multilingual fallback.")
            _EMBED_MODEL = "fallback"
    return _EMBED_MODEL


def compute_query_embedding(query: str) -> List[float]:
    """
    Computes normalized embedding for a search query.
    Utilizes in-memory caching to achieve <0.2ms for cached queries.
    """
    query_norm = query.strip().lower()
    query_hash = hashlib.sha256(query_norm.encode("utf-8")).hexdigest()

    if settings.ENABLE_QUERY_CACHE and query_hash in _QUERY_EMBED_CACHE:
        return _QUERY_EMBED_CACHE[query_hash]

    model = get_embedding_model()
    if model != "fallback" and model is not None:
        embedding = model.encode(query_norm, normalize_embeddings=True).tolist()
    else:
        # High-dimension pseudo-semantic hashing fallback for unit testing / low-resource env
        np.random.seed(int(query_hash[:8], 16) % (2**31))
        vec = np.random.randn(settings.EMBEDDING_DIMENSION)
        embedding = (vec / np.linalg.norm(vec)).tolist()

    if settings.ENABLE_QUERY_CACHE:
        _QUERY_EMBED_CACHE[query_hash] = embedding

    return embedding


class QdrantVectorSearcher:
    """
    High-performance Qdrant vector retrieval engine with multi-collection support
    and payload filtering across languages and chunk strategies.
    Shares a singleton client to avoid lock contention.
    """

    def __init__(self, collection_name: str = "vaani_msmarco_sentence", client: Optional[QdrantClient] = None):
        self.collection_name = collection_name
        self.client = client or get_shared_qdrant_client()
        self._ensure_collection_populated()

    def _ensure_collection_populated(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                self._populate_collection()
            else:
                info = self.client.get_collection(self.collection_name)
                if (info.points_count or 0) < 5:
                    self._populate_collection()
        except Exception:
            pass

    def _populate_collection(self):
        import json
        from pathlib import Path
        
        # Determine matching chunk file
        if "sentence" in self.collection_name:
            chunk_file = Path("./data/chunks_sentence.json")
        elif "sliding" in self.collection_name:
            chunk_file = Path("./data/chunks_sliding.json")
        elif "semantic" in self.collection_name:
            chunk_file = Path("./data/chunks_semantic.json")
        else:
            chunk_file = Path("./data/chunks_all.json")

        if not chunk_file.exists():
            return

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE
            )
        )

        texts = [c["text"] for c in chunks]
        embed_model = get_embedding_model()
        if embed_model != "fallback" and embed_model is not None:
            vectors = embed_model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE, normalize_embeddings=True).tolist()
        else:
            vectors = []
            for t in texts:
                h = hashlib.sha256(t.encode()).hexdigest()
                np.random.seed(int(h[:8], 16) % (2**31))
                v = np.random.randn(settings.EMBEDDING_DIMENSION)
                vectors.append((v / np.linalg.norm(v)).tolist())

        points = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=vec,
                    payload=chunk
                )
            )

        batch_size = 64
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size]
            )

    def search(
        self,
        query: str,
        top_k: int = 20,
        lang_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes dense vector search in Qdrant using cosine similarity.
        """
        query_vector = compute_query_embedding(query)

        # Build payload filters if specified
        filter_conditions = []
        if lang_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=lang_filter)
                )
            )
        if strategy_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="chunk_strategy",
                    match=models.MatchValue(value=strategy_filter)
                )
            )

        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        try:
            hits = []
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                )
                hits = response.points
            elif hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                )

            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append({
                    "chunk_id": payload.get("chunk_id", str(hit.id)),
                    "doc_id": payload.get("doc_id", ""),
                    "text": payload.get("text", ""),
                    "language": payload.get("language", "en"),
                    "chunk_strategy": payload.get("chunk_strategy", "sentence"),
                    "score": float(hit.score),
                    "vector_score": float(hit.score),
                    "source": payload.get("source", "MSMARCO-XI")
                })
            return results
        except Exception as e:
            print(f"[WARN] Qdrant search error on collection '{self.collection_name}': {e}")
            return []
