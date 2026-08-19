import json
from pathlib import Path
from backend.retrieval.bm25_search import IndicBM25Searcher
from ingestion.chunk_dataset import (
    CHUNKS_SENTENCE_PATH,
    CHUNKS_SLIDING_PATH,
    CHUNKS_SEMANTIC_PATH,
    CHUNKS_ALL_PATH,
    process_and_chunk_corpus
)


def build_all_bm25_indices():
    """
    Builds and persists BM25 indices offline for each chunk strategy representation.
    """
    if not CHUNKS_ALL_PATH.exists():
        process_and_chunk_corpus()

    strategies = [
        ("sentence", CHUNKS_SENTENCE_PATH),
        ("sliding_window", CHUNKS_SLIDING_PATH),
        ("semantic", CHUNKS_SEMANTIC_PATH),
        ("combined", CHUNKS_ALL_PATH),
    ]

    for name, path in strategies:
        print(f"[INFO] Building BM25 index for '{name}' from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        searcher = IndicBM25Searcher(index_name=name)
        searcher.build_index(chunks)
        print(f"[SUCCESS] BM25 index '{name}' built successfully ({len(chunks)} passages).")


if __name__ == "__main__":
    build_all_bm25_indices()
