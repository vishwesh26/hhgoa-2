import numpy as np
from typing import List, Dict, Any, Optional
from backend.chunking.sentence_chunker import SentenceChunker


class SemanticChunker:
    """
    Groups sentences into topical semantic chunks based on embedding similarity drops
    between consecutive sentences. Falls back to sentence grouping if embedding model is not provided.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        max_chunk_words: int = 250,
        min_chunk_words: int = 25,
        embedder=None
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_words = max_chunk_words
        self.min_chunk_words = min_chunk_words
        self.sentence_splitter = SentenceChunker(sentences_per_chunk=1)
        self.embedder = embedder

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        sentences = self.sentence_splitter.split_into_sentences(text)
        if not sentences:
            return []

        if len(sentences) <= 2 or self.embedder is None:
            # Group into standard semantic blocks
            return self._heuristic_semantic_chunk(sentences, metadata)

        try:
            # Batch encode sentences
            embeddings = self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            
            chunks = []
            current_sentences = [sentences[0]]
            current_word_count = len(sentences[0].split())

            for i in range(1, len(sentences)):
                sim = float(np.dot(embeddings[i - 1], embeddings[i]))
                sent_words = len(sentences[i].split())

                # Split if similarity drops below threshold or chunk exceeds max words
                if (sim < self.similarity_threshold and current_word_count >= self.min_chunk_words) or (current_word_count + sent_words > self.max_chunk_words):
                    chunk_id = f"{metadata.get('doc_id', 'doc')}_sem_{len(chunks)}"
                    chunks.append({
                        **metadata,
                        "chunk_id": chunk_id,
                        "chunk_strategy": "semantic",
                        "sentence_count": len(current_sentences),
                        "word_count": current_word_count,
                        "text": " ".join(current_sentences)
                    })
                    current_sentences = [sentences[i]]
                    current_word_count = sent_words
                else:
                    current_sentences.append(sentences[i])
                    current_word_count += sent_words

            if current_sentences:
                chunk_id = f"{metadata.get('doc_id', 'doc')}_sem_{len(chunks)}"
                chunks.append({
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_strategy": "semantic",
                    "sentence_count": len(current_sentences),
                    "word_count": current_word_count,
                    "text": " ".join(current_sentences)
                })

            return chunks
        except Exception:
            return self._heuristic_semantic_chunk(sentences, metadata)

    def _heuristic_semantic_chunk(self, sentences: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        current = []
        current_words = 0

        for sent in sentences:
            words = len(sent.split())
            if current_words + words > self.max_chunk_words and current:
                chunk_id = f"{metadata.get('doc_id', 'doc')}_sem_{len(chunks)}"
                chunks.append({
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_strategy": "semantic",
                    "sentence_count": len(current),
                    "word_count": current_words,
                    "text": " ".join(current)
                })
                current = [sent]
                current_words = words
            else:
                current.append(sent)
                current_words += words

        if current:
            chunk_id = f"{metadata.get('doc_id', 'doc')}_sem_{len(chunks)}"
            chunks.append({
                **metadata,
                "chunk_id": chunk_id,
                "chunk_strategy": "semantic",
                "sentence_count": len(current),
                "word_count": current_words,
                "text": " ".join(current)
            })

        return chunks
