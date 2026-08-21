"""
semantic.py - Practical embedding & paragraph-level semantic boundary chunker.
"""
import re
from typing import List, Dict, Any
from .base import BaseChunker, DocumentChunk

class SemanticChunker(BaseChunker):
    def __init__(self, target_chunk_size: int = 150, min_chunk_size: int = 40):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self._sent_re = re.compile(r'([.?!।॥\n\r]+)')

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk(self, text: str, document_id: str, language: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        
        # 1. First attempt paragraph splitting
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 10]
        
        # If single block, group sentences into coherent topic chunks
        if len(paragraphs) <= 1:
            raw_parts = self._sent_re.split(text)
            sentences = []
            curr = ""
            for p in raw_parts:
                if self._sent_re.match(p):
                    curr += p
                    if len(curr.strip()) > 10:
                        sentences.append(curr.strip())
                        curr = ""
                else:
                    curr += p
            if curr.strip():
                sentences.append(curr.strip())

            # Group sentences up to target_chunk_size words
            paragraphs = []
            accum = []
            accum_words = 0
            for s in sentences:
                s_words = len(s.split())
                if accum_words + s_words > self.target_chunk_size and accum_words >= self.min_chunk_size:
                    paragraphs.append(" ".join(accum))
                    accum = [s]
                    accum_words = s_words
                else:
                    accum.append(s)
                    accum_words += s_words
            if accum:
                paragraphs.append(" ".join(accum))

        chunks = []
        for idx, p in enumerate(paragraphs):
            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}_sem_{idx}",
                document_id=document_id,
                text=p,
                language=language,
                chunk_strategy=self.strategy_name,
                source=metadata.get("source", "MSMARCO-XI"),
                metadata={**metadata, "semantic_index": idx, "total_semantic_chunks": len(paragraphs), "word_count": len(p.split())}
            ))

        return chunks
