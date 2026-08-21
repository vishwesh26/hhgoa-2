"""
passage.py - Passage-level chunker preserving candidate passages as complete retrieval units.
"""
from typing import List, Dict, Any, Optional
from .base import BaseChunker, DocumentChunk

class PassageChunker(BaseChunker):
    """
    Passage-level chunker that preserves the entire candidate passage as a single 
    coherent retrieval unit without fragmenting entity-definition relationships.
    """
    @property
    def strategy_name(self) -> str:
        return "passage"

    def chunk(
        self,
        text: str,
        document_id: str,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        clean_text = text.strip()
        if not clean_text:
            return []

        chunk_id = f"{document_id}_passage_0"
        chunk_meta = dict(metadata or {})
        chunk_meta["chunk_strategy"] = "passage"
        chunk_meta["chunk_index"] = 0
        chunk_meta["total_chunks"] = 1

        return [
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=clean_text,
                language=language,
                chunk_strategy="passage",
                source=chunk_meta.get("source", "MSMARCO-XI"),
                metadata=chunk_meta
            )
        ]
