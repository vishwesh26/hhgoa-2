from fastapi import APIRouter
from backend.config import settings

router = APIRouter(tags=["Health"])


@router.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "models": {
            "stt": settings.SARVAM_STT_MODEL,
            "llm": settings.GEMINI_MODEL,
            "embedding": settings.EMBEDDING_MODEL_NAME,
            "reranker": settings.RERANKER_MODEL_NAME
        },
        "features": {
            "multilingual": ["en", "hi", "mr", "code-mixed"],
            "chunk_strategies": ["sentence", "sliding_window", "semantic", "hierarchical"],
            "hybrid_retrieval": ["qdrant_vector", "indic_bm25", "rrf_fusion", "flashrank_rerank"],
            "guardrails": ["prompt_injection", "confidence_gate", "grounding_verifier"]
        }
    }
