import re
from typing import List, Dict, Any


class SentenceChunker:
    """
    Language-aware sentence boundary chunker supporting English (. ? !),
    Hindi (।), Marathi (।), and mixed scripts.
    Groups sentences into clean factual chunks.
    """

    def __init__(self, sentences_per_chunk: int = 2, min_chunk_words: int = 5):
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_words = min_chunk_words
        # Regex matching sentence boundaries across Latin and Indic scripts
        self.sentence_end_pattern = re.compile(r"([.?!।॥]+[\s\n]+|[\n\r]+)")

    def split_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        
        # Split on sentence terminals while retaining coherence
        raw_parts = self.sentence_end_pattern.split(text)
        sentences = []
        current = ""

        for part in raw_parts:
            if not part:
                continue
            current += part
            if self.sentence_end_pattern.search(part):
                cleaned = current.strip()
                if len(cleaned.split()) >= 2:
                    sentences.append(cleaned)
                current = ""

        if current.strip() and len(current.strip().split()) >= 2:
            sentences.append(current.strip())

        if not sentences and text.strip():
            sentences = [text.strip()]

        return sentences

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        sentences = self.split_into_sentences(text)
        chunks = []

        for i in range(0, len(sentences), self.sentences_per_chunk):
            group = sentences[i : i + self.sentences_per_chunk]
            chunk_text = " ".join(group).strip()
            word_count = len(chunk_text.split())

            if word_count < self.min_chunk_words and chunks:
                # Merge small trailing fragments with the previous chunk
                chunks[-1]["text"] += " " + chunk_text
                chunks[-1]["word_count"] = len(chunks[-1]["text"].split())
                continue

            chunk_id = f"{metadata.get('doc_id', 'doc')}_sent_{len(chunks)}"
            chunk_meta = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_strategy": "sentence",
                "sentence_count": len(group),
                "word_count": word_count,
                "text": chunk_text
            }
            chunks.append(chunk_meta)

        return chunks
