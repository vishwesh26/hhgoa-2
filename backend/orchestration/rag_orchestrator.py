import asyncio
import time
from typing import Dict, Any, Optional, List
from backend.config import settings
from backend.orchestration.telemetry import LatencyTracker
from backend.orchestration.query_analyzer import QueryAnalyzer
from backend.retrieval.vector_search import QdrantVectorSearcher
from backend.retrieval.bm25_search import IndicBM25Searcher
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker
from backend.generation.llm_generator import GroundedLLMGenerator
from backend.guardrails.injection_filter import InjectionFilter
from backend.guardrails.confidence_check import ConfidenceGuardrail
from backend.guardrails.grounding_verifier import GroundingVerifier
from backend.generation.extractive_generator import DatasetAnswerExtractor
from backend.voice.sarvam_client import SarvamSTTClient
from backend.voice.audio_processor import AudioProcessor


class RAGOrchestrator:
    """
    Master Orchestration Harness for VAANI Multilingual Adaptive Voice RAG.
    Coordinates speech transcription, query analysis, adaptive parallel retrieval,
    fusion, reranking, grounded generation, and guardrail verification.
    """

    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.injection_filter = InjectionFilter()
        self.confidence_guard = ConfidenceGuardrail()
        self.grounding_verifier = GroundingVerifier()
        self.reranker = CrossEncoderReranker(top_k=settings.FINAL_CONTEXT_K)
        self.llm_generator = GroundedLLMGenerator()
        self.dataset_extractor = DatasetAnswerExtractor()
        self.sarvam_client = SarvamSTTClient()
        self.audio_processor = AudioProcessor()

        # Cache of initialized searchers per collection
        self.vector_searchers = {
            "passage": QdrantVectorSearcher(f"{settings.QDRANT_COLLECTION_PREFIX}_passage"),
            "sliding": QdrantVectorSearcher(f"{settings.QDRANT_COLLECTION_PREFIX}_sliding"),
            "semantic": QdrantVectorSearcher(f"{settings.QDRANT_COLLECTION_PREFIX}_semantic"),
            "sentence": QdrantVectorSearcher(f"{settings.QDRANT_COLLECTION_PREFIX}_sentence"),
            "combined": QdrantVectorSearcher(f"{settings.QDRANT_COLLECTION_PREFIX}_combined"),
        }
        self.bm25_searchers = {
            "passage": IndicBM25Searcher("passage"),
            "sliding": IndicBM25Searcher("sliding"),
            "semantic": IndicBM25Searcher("semantic"),
            "sentence": IndicBM25Searcher("sentence"),
            "combined": IndicBM25Searcher("combined"),
        }

    async def execute_rag(self, query: str, stt_latency_ms: float = 0.0, transcript_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tracker = LatencyTracker()
        if stt_latency_ms > 0:
            tracker.record_stage("stt", stt_latency_ms)

        # 1. Prompt Injection Sanitization
        with tracker.measure("inputValidation"):
            is_safe, sanitized_query = self.injection_filter.check_input(query)
            if not is_safe:
                return {
                    "answer": "Potential security or prompt injection attempt detected. Request refused.",
                    "transcript": query,
                    "language": "en",
                    "codeMixed": False,
                    "retrievalStrategy": "blocked",
                    "chunkStrategy": "none",
                    "sources": [],
                    "confidence": 0.0,
                    "grounded": False,
                    "refused": True,
                    "latency": tracker.get_summary()
                }

        # 2. Query Understanding & Adaptive Strategy Selection
        with tracker.measure("queryAnalysis"):
            analysis = self.query_analyzer.analyze(sanitized_query)
            lang = analysis["language"]
            is_code_mixed = analysis["is_code_mixed"]
            query_type = analysis["query_type"]
            strategy = analysis["adaptive_strategy"]
            target_collection = strategy["target_collection"]
            vector_weight = strategy["vector_weight"]
            bm25_weight = strategy["bm25_weight"]

        # 3. Parallel Retrieval: Qdrant Vector + BM25 Lexical
        vec_searcher = self.vector_searchers.get(target_collection, self.vector_searchers["combined"])
        bm25_searcher = self.bm25_searchers.get(target_collection, self.bm25_searchers["combined"])
        bm25_combined = self.bm25_searchers.get("combined", self.bm25_searchers["sentence"])

        # Run Qdrant and BM25 concurrently using asyncio with individual timers
        loop = asyncio.get_event_loop()

        def _timed_vec():
            t0 = time.perf_counter()
            res = vec_searcher.search(sanitized_query, strategy["top_k"])
            return res, (time.perf_counter() - t0) * 1000.0

        def _timed_bm25():
            t0 = time.perf_counter()
            res1 = bm25_searcher.search(sanitized_query, strategy["top_k"])
            res2 = bm25_combined.search(sanitized_query, strategy["top_k"])
            return (res1, res2), (time.perf_counter() - t0) * 1000.0

        (vector_results, vec_lat_ms), ((bm25_results, bm25_comb_res), bm25_lat_ms) = await asyncio.gather(
            loop.run_in_executor(None, _timed_vec),
            loop.run_in_executor(None, _timed_bm25)
        )
        tracker.record_stage("vectorSearch", vec_lat_ms)
        tracker.record_stage("bm25Search", bm25_lat_ms)

        # Merge BM25 specific and combined results
        seen_bm25_chunks = set()
        merged_bm25 = []
        for r in (bm25_results or []) + (bm25_comb_res or []):
            cid = r.get("chunk_id")
            if cid and cid not in seen_bm25_chunks:
                seen_bm25_chunks.add(cid)
                merged_bm25.append(r)

        # 4. Reciprocal Rank Fusion (RRF)
        with tracker.measure("fusion"):
            fused_candidates = reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=merged_bm25,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                rrf_k=settings.RRF_K,
                top_k=settings.VECTOR_TOP_K
            )

        # 5. Cross-Encoder Reranking
        with tracker.measure("reranking"):
            reranked_sources = self.reranker.rerank(sanitized_query, fused_candidates)

        # 6. Retrieval Confidence Guardrail
        is_confident, confidence_score, conf_message = self.confidence_guard.evaluate_confidence(
            reranked_sources,
            query=sanitized_query
        )
        if not is_confident:
            refusal_text = self._get_refusal_text(lang)
            return {
                "answer": refusal_text,
                "transcript": query,
                "language": lang,
                "codeMixed": is_code_mixed,
                "queryType": query_type,
                "retrievalStrategy": "adaptive_hybrid",
                "chunkStrategy": target_collection,
                "sources": reranked_sources[:3],
                "confidence": confidence_score,
                "grounded": True,
                "refused": True,
                "refusalReason": conf_message,
                "latency": tracker.get_summary()
            }

        # 7. Answer Generation via Grounded LLM
        t_gen_start = time.perf_counter()
        # Sanitize any injection patterns in retrieved texts
        for s in reranked_sources:
            s["text"] = self.injection_filter.sanitize_retrieved_passage(s.get("text", ""))

        answer_text, gen_lat_ms, model_used = await self.llm_generator.generate_grounded_answer(
            query=sanitized_query,
            retrieved_sources=reranked_sources,
            detected_lang=lang
        )
        tracker.record_stage("generation", gen_lat_ms)

        # 8. Grounding Verification
        with tracker.measure("grounding"):
            is_grounded, support_score, ground_msg = self.grounding_verifier.verify(answer_text, reranked_sources)

        return {
            "answer": answer_text,
            "transcript": query,
            "language": lang,
            "codeMixed": is_code_mixed,
            "queryType": query_type,
            "retrievalStrategy": "adaptive_hybrid",
            "chunkStrategy": target_collection,
            "strategyRationale": strategy.get("rationale", ""),
            "sources": [
                {
                    "chunkId": s.get("chunk_id", ""),
                    "docId": s.get("doc_id", ""),
                    "language": s.get("language", "en"),
                    "chunkStrategy": s.get("chunk_strategy", "standard"),
                    "score": round(s.get("score", 0.0), 3),
                    "text": s.get("text", "")
                }
                for s in reranked_sources[:settings.FINAL_CONTEXT_K]
            ],
            "confidence": confidence_score,
            "grounded": is_grounded,
            "groundingScore": support_score,
            "modelUsed": model_used,
            "refused": False,
            "latency": tracker.get_summary()
        }

    async def execute_voice_rag(self, audio_bytes: bytes, filename: str = "voice.wav") -> Dict[str, Any]:
        # Validate Audio
        is_valid, msg = self.audio_processor.validate_audio(audio_bytes)
        if not is_valid:
            return {
                "answer": msg,
                "transcript": "",
                "language": "en",
                "codeMixed": False,
                "sources": [],
                "confidence": 0.0,
                "grounded": False,
                "refused": True,
                "latency": {"totalLatencyMs": 0}
            }

        # Transcribe with Sarvam
        transcript, lang, stt_latency, meta = await self.sarvam_client.transcribe(audio_bytes, filename=filename)
        if not transcript:
            err_details = meta.get("error", "No speech recognized. Please speak clearly into your microphone and try again.")
            return {
                "answer": f"{err_details}",
                "transcript": "",
                "language": lang,
                "codeMixed": False,
                "sources": [],
                "confidence": 0.0,
                "grounded": False,
                "refused": True,
                "latency": {"stt": stt_latency, "totalLatencyMs": stt_latency}
            }

        return await self.execute_rag(transcript, stt_latency_ms=stt_latency, transcript_meta=meta)

    def _get_refusal_text(self, lang: str) -> str:
        if lang == "hi":
            return "मुझे उपलब्ध ज्ञानकोष में इसका विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"
        elif lang == "mr":
            return "मला उपलब्ध माहितीच्या आधारे या प्रश्नाचे विश्वसनीय उत्तर देण्यासाठी पुरेशी माहिती मिळाली नाही."
        return "I couldn't find sufficient information in the provided knowledge base to answer that reliably."
