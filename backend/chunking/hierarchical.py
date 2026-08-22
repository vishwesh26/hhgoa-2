from typing import List, Dict, Any, Optional
from backend.chunking.sentence_chunker import SentenceChunker
from backend.chunking.sliding_chunker import SlidingWindowChunker
from backend.chunking.semantic_chunker import SemanticChunker


class MultiStrategyChunker:
    """
    Unified multi-strategy chunking orchestrator.
    Generates sentence-based, sliding-window, and semantic representations
    enriched with full metadata for indexing into specialized collections.
    """

    def __init__(self, embedder=None):
        self.sentence_chunker = SentenceChunker(sentences_per_chunk=2, min_chunk_words=4)
        self.sliding_chunker = SlidingWindowChunker(window_size=60, overlap=15)
        self.semantic_chunker = SemanticChunker(max_chunk_words=70, min_chunk_words=15, embedder=embedder)

    def process_document(
        self,
        doc_id: str,
        text: str,
        language: str,
        title: Optional[str] = None,
        source: str = "MSMARCO-XI",
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        base_meta = {
            "doc_id": doc_id,
            "language": language,
            "title": title or "",
            "source": source,
            **(extra_meta or {})
        }

        sentence_chunks = self.sentence_chunker.chunk(text, metadata=base_meta)
        sliding_chunks = self.sliding_chunker.chunk(text, metadata=base_meta)
        semantic_chunks = self.semantic_chunker.chunk(text, metadata=base_meta)

        return {
            "sentence": sentence_chunks,
            "sliding_window": sliding_chunks,
            "semantic": semantic_chunks
        }
