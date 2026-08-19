from typing import List, Dict, Any


class SlidingWindowChunker:
    """
    Sliding window chunker with configurable word/token window size and step overlap.
    Preserves context across sentence boundaries.
    """

    def __init__(self, window_size: int = 120, overlap: int = 40):
        self.window_size = window_size
        self.overlap = overlap
        self.step = max(1, window_size - overlap)

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        words = text.strip().split()
        if not words:
            return []

        if len(words) <= self.window_size:
            chunk_id = f"{metadata.get('doc_id', 'doc')}_sliding_0"
            return [{
                **metadata,
                "chunk_id": chunk_id,
                "chunk_strategy": "sliding_window",
                "word_count": len(words),
                "start_idx": 0,
                "end_idx": len(words),
                "text": " ".join(words)
            }]

        chunks = []
        for i in range(0, len(words), self.step):
            window_words = words[i : i + self.window_size]
            if not window_words:
                break
            
            # Avoid tiny trailing chunk if less than half step size
            if len(window_words) < (self.step // 2) and chunks:
                break

            chunk_id = f"{metadata.get('doc_id', 'doc')}_sliding_{len(chunks)}"
            chunks.append({
                **metadata,
                "chunk_id": chunk_id,
                "chunk_strategy": "sliding_window",
                "word_count": len(window_words),
                "start_idx": i,
                "end_idx": i + len(window_words),
                "text": " ".join(window_words)
            })

            if i + self.window_size >= len(words):
                break

        return chunks
