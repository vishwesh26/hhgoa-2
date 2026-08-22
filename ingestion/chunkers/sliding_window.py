"""
sliding_window.py - Sliding window chunker with configurable window size and overlap.
"""
from typing import List, Dict, Any
from .base import BaseChunker, DocumentChunk

class SlidingWindowChunker(BaseChunker):
    def __init__(self, window_size: int = 55, overlap: int = 15):
        if overlap >= window_size:
            raise ValueError("Overlap must be strictly smaller than window_size.")
        self.window_size = window_size
        self.overlap = overlap
        self.step = window_size - overlap

    @property
    def strategy_name(self) -> str:
        return "sliding"

    def chunk(self, text: str, document_id: str, language: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        words = text.strip().split()
        
        if len(words) <= self.window_size:
            return [DocumentChunk(
                chunk_id=f"{document_id}_slide_0",
                document_id=document_id,
                text=text.strip(),
                language=language,
                chunk_strategy=self.strategy_name,
                source=metadata.get("source", "MSMARCO-XI"),
                metadata={**metadata, "window_index": 0, "total_windows": 1, "word_count": len(words)}
            )]

        chunks = []
        idx = 0
        w_idx = 0
        while idx < len(words):
            window_words = words[idx : idx + self.window_size]
            chunk_text = " ".join(window_words)
            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}_slide_{w_idx}",
                document_id=document_id,
                text=chunk_text,
                language=language,
                chunk_strategy=self.strategy_name,
                source=metadata.get("source", "MSMARCO-XI"),
                metadata={**metadata, "window_index": w_idx, "word_count": len(window_words)}
            ))
            w_idx += 1
            idx += self.step
            # If remaining words are fewer than overlap and we already made chunks, break
            if idx + self.overlap >= len(words) and len(words) - idx < (self.window_size // 3):
                break

        for c in chunks:
            c.metadata["total_windows"] = len(chunks)

        return chunks
