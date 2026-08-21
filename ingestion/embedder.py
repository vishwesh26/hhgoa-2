"""
embedder.py - Batched multilingual embedding generator supporting FastEmbed ONNX and SentenceTransformers.
"""
import logging
import numpy as np
from typing import List, Optional
from backend.config import settings

logger = logging.getLogger(__name__)

class MultilingualEmbedder:
    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._mode = "uninitialized"
        self._init_model()

    def _init_model(self):
        # 1. Try FastEmbed ONNX
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            self._mode = "fastembed"
            logger.info(f"Initialized FastEmbed ONNX Multilingual Model: {self.model_name}")
            return
        except Exception as e_fast:
            logger.warning(f"FastEmbed init failed: {e_fast}. Trying SentenceTransformer...")

        # 2. Try SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._mode = "sentence_transformers"
            logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            return
        except Exception as e_st:
            logger.error(f"SentenceTransformer load failed: {e_st}")

        # FAIL FAST
        raise RuntimeError(
            f"Embedding model '{self.model_name}' is NOT AVAILABLE. "
            f"Production RAG requires a valid multilingual embedding model. (FastEmbed/SentenceTransformers)"
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates normalized dense vector embeddings for a list of texts in batches.
        """
        if not texts:
            return []

        if self._mode == "fastembed" and self._model is not None:
            raw_embeddings = list(self._model.embed(texts, batch_size=self.batch_size))
            vectors = []
            for v in raw_embeddings:
                norm = np.linalg.norm(v)
                vectors.append((v / norm).tolist() if norm > 0 else v.tolist())
            return vectors

        if self._mode == "sentence_transformers" and self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()

        raise RuntimeError("No valid embedding model loaded.")

    def embed_query(self, query: str) -> List[float]:
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * self.dimension
