import base64
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from backend.api.routes_rag import get_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/voice", tags=["Voice"])


class Base64VoiceRequest(BaseModel):
    audio_base64: str
    filename: Optional[str] = "recording.wav"


@router.post("/query")
@router.post("/query/")
async def voice_query_upload(
    file: UploadFile = File(...),
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Accepts direct multipart audio upload (WAV/WebM/MP3) for Sarvam STT and Adaptive RAG.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        
        response = await orchestrator.execute_voice_rag(audio_bytes, filename=file.filename or "audio.wav")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Voice RAG pipeline failed during audio upload processing")
        raise HTTPException(status_code=500, detail=f"Voice RAG pipeline failed: {str(e)}")


@router.post("/query-base64")
async def voice_query_base64(
    request: Base64VoiceRequest,
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Accepts Base64 encoded audio from client Web Audio recorder.
    """
    try:
        # Strip potential data URL prefix
        data = request.audio_base64
        if "," in data:
            data = data.split(",")[1]

        audio_bytes = base64.b64decode(data)
        response = await orchestrator.execute_voice_rag(audio_bytes, filename=request.filename or "audio.wav")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Voice RAG pipeline failed during base64 processing")
        raise HTTPException(status_code=500, detail=f"Voice base64 processing failed: {str(e)}")


@router.post("/transcribe")
async def voice_transcribe_only(
    file: UploadFile = File(...),
    orchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Direct Speech-to-Text transcription without executing RAG retrieval.
    """
    try:
        audio_bytes = await file.read()
        transcript, lang, latency, meta = await orchestrator.sarvam_client.transcribe(
            audio_bytes, filename=file.filename or "audio.wav"
        )
        return {
            "transcript": transcript,
            "language": lang,
            "stt_latency_ms": latency,
            "meta": meta
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("STT transcription failed")
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {str(e)}")

