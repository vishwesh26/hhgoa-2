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
        with _INIT_LOCK:
            if _SHARED_QDRANT_CLIENT is None:
                try:
                    import psutil
                    rss_before = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
                    print(f"[MEMORY] Before Qdrant client init: {rss_before} MB")
                except Exception:
                    pass

                print(f"[INFO] Initializing Qdrant client (embedded={settings.QDRANT_USE_EMBEDDED})...")
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

                try:
                    import psutil
                    rss_after = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
                    print(f"[MEMORY] After Qdrant client init: {rss_after} MB")
                except Exception:
                    pass

    return _SHARED_QDRANT_CLIENT


import threading

# Thread lock for safe single-initialization
_INIT_LOCK = threading.Lock()


def get_embedding_model():
    """
    Loads high-performance multilingual embedding model as a thread-safe singleton.
    Configures ONNX Runtime with conservative single-thread execution for low-memory environments.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _INIT_LOCK:
            if _EMBED_MODEL is None:
                try:
                    import psutil
                    rss_before = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
                    print(f"[MEMORY] Before embedding model init: {rss_before} MB")
                except Exception:
                    pass

                print(f"[INFO] Initializing FastEmbed ONNX Multilingual Model: {settings.EMBEDDING_MODEL_NAME}...")
                try:
                    from fastembed import TextEmbedding
                    # threads=1 conserves thread pool allocations on low-memory Render containers
                    _EMBED_MODEL = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME, threads=1)
                    print(f"[HEALTH] Embedding Model READY (FastEmbed ONNX, Dim: {settings.EMBEDDING_DIMENSION})")
                except Exception as e_fast:
                    print(f"[WARN] FastEmbed init exception: {e_fast}")
                    try:
                        from sentence_transformers import SentenceTransformer
                        _EMBED_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                    except Exception as e_st:
                        raise RuntimeError(
                            f"Embedding model unavailable (FastEmbed: {e_fast}, SentenceTransformer: {e_st})"
                        )

                try:
                    import psutil
                    rss_after = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
                    print(f"[MEMORY] After embedding model init: {rss_after} MB")
                except Exception:
                    pass

    return _EMBED_MODEL


def check_embedding_health() -> Dict[str, Any]:
    """Startup health check for embedding engine."""
    try:
        model = get_embedding_model()
        test_vec = compute_query_embedding("स्वास्थ्य परीक्षण")
        return {
            "status": "READY",
            "model_name": settings.EMBEDDING_MODEL_NAME,
            "dimension": len(test_vec),
            "engine": type(model).__name__,
            "error": None
        }
    except Exception as e:
        return {
            "status": "NOT AVAILABLE",
            "model_name": settings.EMBEDDING_MODEL_NAME,
            "dimension": settings.EMBEDDING_DIMENSION,
            "engine": None,
            "error": str(e)
        }


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
    if hasattr(model, "embed"):
        # FastEmbed interface
        embeddings = list(model.embed([query_norm]))
        vec = embeddings[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embedding = vec.tolist()
    elif hasattr(model, "encode"):
        # SentenceTransformer interface
        embedding = model.encode(query_norm, normalize_embeddings=True).tolist()
    else:
        raise RuntimeError("Loaded embedding model has unrecognized interface.")

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
        if hasattr(embed_model, "embed"):
            raw_vecs = list(embed_model.embed(texts))
            vectors = []
            for v in raw_vecs:
                norm = np.linalg.norm(v)
                vectors.append((v / norm).tolist() if norm > 0 else v.tolist())
        elif hasattr(embed_model, "encode"):
            vectors = embed_model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE, normalize_embeddings=True).tolist()
        else:
            raise RuntimeError("No valid embedding model available to populate collection.")

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
                    "doc_id": payload.get("document_id", payload.get("doc_id", "")),
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
