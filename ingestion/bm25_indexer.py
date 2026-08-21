"""
bm25_indexer.py - Persistent BM25 lexical indexer supporting English, Hindi, and Marathi.
"""
import os
import pickle
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from .clean_indic import indic_tokenize
from .chunkers.base import DocumentChunk

logger = logging.getLogger(__name__)

class BM25Indexer:
    def __init__(self, index_dir: str = "./data/bm25_indices"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

    def get_index_path(self, strategy: str) -> str:
        return os.path.join(self.index_dir, f"bm25_{strategy}.pkl")

    def build_and_save(
        self,
        strategy: str,
        chunks: List[DocumentChunk]
    ) -> int:
        """
        Builds a persistent BM25 index over the provided chunks.
        """
        if not chunks:
            return 0

        corpus_tokens = []
        doc_store = []

        for chunk in chunks:
            tokens = indic_tokenize(chunk.text, remove_stopwords=True, lang=chunk.language)
            corpus_tokens.append(tokens)
            doc_store.append(chunk.to_dict())

        bm25 = BM25Okapi(corpus_tokens)

        data = {
            "bm25": bm25,
            "doc_store": doc_store,
            "strategy": strategy,
            "total_docs": len(doc_store)
        }

        save_path = self.get_index_path(strategy)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index for '{strategy}' ({len(doc_store)} docs) to {save_path}")
        return len(doc_store)

    def load_index(self, strategy: str) -> Dict[str, Any]:
        save_path = self.get_index_path(strategy)
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"BM25 index not found: {save_path}")
        with open(save_path, "rb") as f:
            return pickle.load(f)
