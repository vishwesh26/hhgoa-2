"""
sentence.py - Language-aware sentence boundary chunker supporting English, Hindi, and Marathi.
"""
import re
from typing import List, Dict, Any
from .base import BaseChunker, DocumentChunk

class SentenceChunker(BaseChunker):
    def __init__(self, min_sentence_len: int = 20):
        self.min_sentence_len = min_sentence_len
        # Split on Latin period, exclamation, question mark, or Devanagari danda (\u0964) and double danda (\u0965)
        self._split_pattern = re.compile(r'([.?!।॥\n\r]+)')

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def chunk(self, text: str, document_id: str, language: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        parts = self._split_pattern.split(text)
        
        sentences = []
        current = ""
        for p in parts:
            if not p:
                continue
            if self._split_pattern.match(p):
                current += p
                if len(current.strip()) >= self.min_sentence_len:
                    sentences.append(current.strip())
                    current = ""
            else:
                current += p
                
        if current.strip() and len(current.strip()) >= self.min_sentence_len:
            sentences.append(current.strip())
        elif current.strip() and sentences:
            sentences[-1] += " " + current.strip()
        elif current.strip():
            sentences.append(current.strip())

        chunks = []
        for idx, s in enumerate(sentences):
            c_id = f"{document_id}_sent_{idx}"
            chunks.append(DocumentChunk(
                chunk_id=c_id,
                document_id=document_id,
                text=s,
                language=language,
                chunk_strategy=self.strategy_name,
                source=metadata.get("source", "MSMARCO-XI"),
                metadata={**metadata, "sentence_index": idx, "total_sentences": len(sentences)}
            ))

        return chunks
