"""
qdrant_indexer.py - Qdrant vector database collection manager and batched indexer.
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config import settings
from .chunkers.base import DocumentChunk

logger = logging.getLogger(__name__)

class QdrantIndexer:
    def __init__(
        self,
        storage_path: Optional[str] = None,
        dimension: Optional[int] = None,
        collection_prefix: Optional[str] = None
    ):
        self.storage_path = storage_path or settings.QDRANT_STORAGE_PATH
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.collection_prefix = collection_prefix or settings.QDRANT_COLLECTION_PREFIX
        os.makedirs(self.storage_path, exist_ok=True)
        self.client = QdrantClient(path=self.storage_path)

    def get_collection_name(self, strategy: str) -> str:
        return f"{self.collection_prefix}_{strategy}"

    def ensure_collection(self, strategy: str):
        col_name = self.get_collection_name(strategy)
        collections = [c.name for c in self.client.get_collections().collections]
        if col_name not in collections:
            logger.info(f"Creating Qdrant collection: {col_name} (dim={self.dimension})")
            self.client.create_collection(
                collection_name=col_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE
                )
            )
            # Create payload indices for fast filtering
            self.client.create_payload_index(
                collection_name=col_name,
                field_name="language",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=col_name,
                field_name="chunk_strategy",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=col_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def upload_chunk_batch(
        self,
        strategy: str,
        chunks: List[DocumentChunk],
        vectors: List[List[float]]
    ) -> int:
        if not chunks or not vectors:
            return 0

        self.ensure_collection(strategy)
        col_name = self.get_collection_name(strategy)

        points = []
        for chunk, vec in zip(chunks, vectors):
            # Generate deterministic UUID from chunk_id
            pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "language": chunk.language,
                "chunk_strategy": chunk.chunk_strategy,
                "source": chunk.source,
                **chunk.metadata
            }
            points.append(models.PointStruct(
                id=pt_id,
                vector=vec,
                payload=payload
            ))

        self.client.upsert(
            collection_name=col_name,
            points=points
        )
        return len(points)

    def get_point_count(self, strategy: str) -> int:
        col_name = self.get_collection_name(strategy)
        try:
            info = self.client.get_collection(col_name)
            return info.points_count or 0
        except Exception:
            return 0
