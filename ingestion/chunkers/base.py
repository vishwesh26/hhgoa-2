"""
base.py - Abstract base class and data model for chunking strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    language: str
    chunk_strategy: str
    source: str = "MSMARCO-XI"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "language": self.language,
            "chunk_strategy": self.chunk_strategy,
            "source": self.source,
            "metadata": self.metadata
        }

class BaseChunker(ABC):
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass

    @abstractmethod
    def chunk(self, text: str, document_id: str, language: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Split document text into a list of DocumentChunk instances.
        """
        pass
