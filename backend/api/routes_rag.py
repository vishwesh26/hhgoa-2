from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Singleton orchestrator (lazy-loaded to avoid slow imports blocking port binding)
_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from backend.orchestration.rag_orchestrator import RAGOrchestrator
        _orchestrator = RAGOrchestrator()
    return _orchestrator


class TextQueryRequest(BaseModel):
    query: str = Field(..., description="User question in English, Hindi, Marathi, or code-mixed speech")
    stream: bool = Field(default=False, description="Whether to stream the answer")


@router.post("/query")
@router.post("/query/")
async def query_rag(
    request: TextQueryRequest,
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Processes an incoming text query through the Adaptive Multilingual RAG pipeline.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = await orchestrator.execute_rag(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")
