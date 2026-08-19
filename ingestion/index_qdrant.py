import json
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config import settings
from backend.retrieval.vector_search import get_embedding_model
from ingestion.chunk_dataset import (
    CHUNKS_SENTENCE_PATH,
    CHUNKS_SLIDING_PATH,
    CHUNKS_SEMANTIC_PATH,
    CHUNKS_ALL_PATH,
    process_and_chunk_corpus
)


def get_qdrant_client() -> QdrantClient:
    storage_path = Path(settings.QDRANT_STORAGE_PATH)
    storage_path.mkdir(parents=True, exist_ok=True)
    if settings.QDRANT_USE_EMBEDDED:
        lock_file = storage_path / ".lock"
        if lock_file.exists():
            try:
                lock_file.unlink()
            except Exception:
                pass
        return QdrantClient(path=settings.QDRANT_STORAGE_PATH)
    else:
        return QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )


def index_chunks_into_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embed_model
):
    print(f"[INFO] Indexing {len(chunks)} points into Qdrant collection '{collection_name}'...")

    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=1000
        )
    )

    # Create payload indexes for fast filtering
    client.create_payload_index(
        collection_name=collection_name,
        field_name="language",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_strategy",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    # Extract texts and compute batch embeddings
    texts = [c["text"] for c in chunks]
    if embed_model != "fallback" and embed_model is not None:
        vectors = embed_model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE, normalize_embeddings=True).tolist()
    else:
        import numpy as np
        import hashlib
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

    # Batch upsert points
    batch_size = 64
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size]
        )

    print(f"[SUCCESS] Indexed {len(points)} vectors into '{collection_name}'.")


def build_all_qdrant_collections():
    """
    Offline builder for all Qdrant vector collections.
    """
    if not CHUNKS_ALL_PATH.exists():
        process_and_chunk_corpus()

    client = get_qdrant_client()
    embed_model = get_embedding_model()

    collections = [
        (f"{settings.QDRANT_COLLECTION_PREFIX}_sentence", CHUNKS_SENTENCE_PATH),
        (f"{settings.QDRANT_COLLECTION_PREFIX}_sliding", CHUNKS_SLIDING_PATH),
        (f"{settings.QDRANT_COLLECTION_PREFIX}_semantic", CHUNKS_SEMANTIC_PATH),
        (f"{settings.QDRANT_COLLECTION_PREFIX}_combined", CHUNKS_ALL_PATH),
    ]

    for col_name, file_path in collections:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        index_chunks_into_qdrant(client, col_name, chunks, embed_model)


if __name__ == "__main__":
    build_all_qdrant_collections()
