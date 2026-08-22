import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # App metadata
    APP_NAME: str = "VAANI Voice-Enabled Multilingual Adaptive RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Sarvam AI STT Credentials & URLs
    SARVAM_API_KEY: str = Field(default="")
    SARVAM_STT_MODEL: str = "saarika:v2.5"
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    SARVAM_TIMEOUT_SECONDS: float = 8.0

    # Google Gemini LLM Settings
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_FALLBACK_MODEL: str = "gemini-1.5-flash"
    LLM_TIMEOUT_SECONDS: float = 8.0
    MAX_GENERATION_TOKENS: int = 256
    TEMPERATURE: float = 0.0
    ANSWER_GENERATION_MODE: str = "extractive"  # 'extractive' (<2ms sub-200ms total), 'generative' (LLM)

    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_PREFIX: str = "msmarco_xi_v2"
    QDRANT_USE_EMBEDDED: bool = True
    QDRANT_STORAGE_PATH: str = "./data/qdrant_storage"

    # Embedding Model Settings (FastEmbed Multilingual ONNX)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_USE_ONNX: bool = True
    EMBEDDING_BATCH_SIZE: int = 64

    # Reranker Settings
    RERANKER_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"
    RERANKER_TOP_K: int = 6
    FINAL_CONTEXT_K: int = 2
    RERANKER_TIMEOUT_SECONDS: float = 1.0

    # Lexical / BM25 Search
    BM25_INDEX_DIR: str = "./data/bm25_indices"
    BM25_TOP_K: int = 10
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # Hybrid Search & RRF Parameters
    VECTOR_TOP_K: int = 10
    RRF_K: int = 60
    DEFAULT_VECTOR_WEIGHT: float = 0.15
    DEFAULT_BM25_WEIGHT: float = 0.85

    # Guardrails & Calibrated Thresholds
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.45
    GROUNDING_VERIFICATION_THRESHOLD: float = 0.55

    # Performance & Caching
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    ENABLE_QUERY_CACHE: bool = True
    CACHE_MAX_SIZE: int = 2000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
