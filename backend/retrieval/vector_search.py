import os
import hashlib
import time
import threading
from typing import List, Dict, Any, Optional
from cachetools import LRUCache
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config import settings

# Limit OpenMP / BLAS threads to 1 to conserve RAM and prevent thread pool explosion on Render
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Global LRU cache for query embeddings: query_hash -> list of floats
_QUERY_EMBED_CACHE = LRUCache(maxsize=settings.CACHE_MAX_SIZE)

# Global embedding model instance (lazy-loaded)
_EMBED_MODEL = None

# Global shared Qdrant client instance
_SHARED_QDRANT_CLIENT = None

# Thread lock for safe single-initialization
_INIT_LOCK = threading.Lock()


def get_shared_qdrant_client() -> QdrantClient:
    global _SHARED_QDRANT_CLIENT
    if _SHARED_QDRANT_CLIENT is None:
        with _INIT_LOCK:
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
    """
    Loads high-performance multilingual embedding model as a thread-safe singleton.
    Configures single-thread execution for low-memory environments.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _INIT_LOCK:
            if _EMBED_MODEL is None:
                os.makedirs(settings.FASTEMBED_CACHE_PATH, exist_ok=True)
                try:
                    from fastembed import TextEmbedding
                    _EMBED_MODEL = TextEmbedding(
                        model_name=settings.EMBEDDING_MODEL_NAME,
                        cache_dir=settings.FASTEMBED_CACHE_PATH,
                        threads=1
                    )
                except Exception as e_fast:
                    try:
                        from sentence_transformers import SentenceTransformer
                        _EMBED_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                    except Exception as e_st:
                        print(f"[WARN] Embedding model init fallback: {e_fast}, {e_st}")
                        _EMBED_MODEL = None

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
    Computes normalized embedding vector for an input query string.
    Employs fast hashing and LRU caching for instant lookup of repeated queries.
    """
    try:
        from ingestion.clean_indic import normalize_indic_text
        query_norm = normalize_indic_text(query)
        if not query_norm:
            return [0.0] * settings.EMBEDDING_DIMENSION

        query_hash = hashlib.sha256(query_norm.encode("utf-8")).hexdigest()
        if settings.ENABLE_QUERY_CACHE and query_hash in _QUERY_EMBED_CACHE:
            return _QUERY_EMBED_CACHE[query_hash]

        model = get_embedding_model()
        if hasattr(model, "embed"):
            embeddings = list(model.embed([query_norm]))
            vec = embeddings[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embedding = vec.tolist()
        elif hasattr(model, "encode"):
            embedding = model.encode(query_norm, normalize_embeddings=True).tolist()
        else:
            return [0.0] * settings.EMBEDDING_DIMENSION

        if settings.ENABLE_QUERY_CACHE:
            _QUERY_EMBED_CACHE[query_hash] = embedding

        return embedding
    except Exception as e:
        print(f"[WARN] Embedding computation fallback: {e}")
        return [0.0] * settings.EMBEDDING_DIMENSION


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

        if len(chunks) > 250:
            chunks = chunks[:250]

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )

            texts = [c["text"] for c in chunks]
            model = get_embedding_model()
            if hasattr(model, "embed"):
                embeddings = list(model.embed(texts, batch_size=settings.EMBEDDING_BATCH_SIZE))
            else:
                embeddings = model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE, normalize_embeddings=True)

            points = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "document_id": chunk["doc_id"],
                            "text": chunk["text"],
                            "language": chunk.get("language", "en"),
                            "chunk_strategy": chunk.get("chunk_strategy", "sentence"),
                            "source": chunk.get("source", "MSMARCO-XI")
                        }
                    )
                )

            batch_size = 100
            for i in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points[i : i + batch_size]
                )
        except Exception as e:
            print(f"[WARN] Error populating collection {self.collection_name}: {e}")

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
        try:
            query_vector = compute_query_embedding(query)
            if not query_vector or sum(query_vector) == 0.0:
                return []

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
            print(f"[WARN] Qdrant search fallback on collection '{self.collection_name}': {e}")
            return []
