import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from ingestion.clean_indic import indic_tokenize, normalize_indic_text
from backend.config import settings


class IndicBM25Searcher:
    """
    High-performance in-memory BM25 lexical search engine
    optimized for Indic languages (Hindi, Marathi, English, and Hinglish).
    Operates in microseconds.
    """

    def __init__(self, index_name: str = "sentence"):
        self.index_name = index_name
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.index_dir = Path(settings.BM25_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / f"bm25_{index_name}.pkl"

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 index over the provided chunks using language-aware tokenization.
        """
        self.corpus_chunks = chunks
        tokenized_corpus = []

        for c in chunks:
            text = c["text"]
            lang = c.get("language", "auto")
            tokens = indic_tokenize(text, remove_stopwords=True, lang=lang)
            tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=settings.BM25_K1,
            b=settings.BM25_B
        )
        self.save()

    def save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "corpus_chunks": self.corpus_chunks,
                "bm25": self.bm25
            }, f)

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.corpus_chunks = data["corpus_chunks"]
                self.bm25 = data["bm25"]
            return True
        except Exception as e:
            print(f"[WARN] Failed to load BM25 index {self.index_name}: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 20,
        lang_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs BM25 lexical search for the query and returns top_k scored results.
        """
        if self.bm25 is None:
            if not self.load():
                return []

        query_tokens = indic_tokenize(query, remove_stopwords=False)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Rank candidate indices
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0

        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                break
            
            chunk = self.corpus_chunks[idx]
            if lang_filter and chunk.get("language") != lang_filter:
                continue

            # Normalized BM25 score in range [0, 1]
            norm_score = min(1.0, score / max_score)
            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "doc_id": chunk.get("doc_id", ""),
                "text": chunk.get("text", ""),
                "language": chunk.get("language", "en"),
                "chunk_strategy": chunk.get("chunk_strategy", self.index_name),
                "bm25_score": score,
                "score": norm_score,
                "source": chunk.get("source", "MSMARCO-XI")
            })

            if len(results) >= top_k:
                break

        return results
