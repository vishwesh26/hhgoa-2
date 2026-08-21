from .base import BaseChunker, DocumentChunk
from .passage import PassageChunker
from .sentence import SentenceChunker
from .sliding_window import SlidingWindowChunker
from .semantic import SemanticChunker

def get_chunker(strategy: str) -> BaseChunker:
    strategies = {
        "passage": PassageChunker(),
        "sentence": SentenceChunker(),
        "sliding": SlidingWindowChunker(),
        "semantic": SemanticChunker()
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Available: {list(strategies.keys())}")
    return strategies[strategy]

__all__ = ["BaseChunker", "DocumentChunk", "PassageChunker", "SentenceChunker", "SlidingWindowChunker", "SemanticChunker", "get_chunker"]

