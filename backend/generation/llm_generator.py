import os
import time
from typing import List, Dict, Any, Tuple
from backend.config import settings
from backend.generation.prompts import STRICT_RAG_SYSTEM_PROMPT, format_generation_prompt


class GroundedLLMGenerator:
    """
    High-performance Grounded LLM Generator leveraging Google Gemini
    (primary: gemini-3.1-pro-preview, secondary: gemini-2.5-flash) with
    strict grounding, anti-injection, and fallback mechanisms.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = settings.GEMINI_MODEL
        self.fallback_model = settings.GEMINI_FALLBACK_MODEL
        self._client_initialized = False
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client_initialized = True
            except Exception as e:
                print(f"[WARN] Gemini client init warning: {e}")

    async def generate_grounded_answer(
        self,
        query: str,
        retrieved_sources: List[Dict[str, Any]],
        detected_lang: str = "en"
    ) -> Tuple[str, float, str]:
        """
        Generates a grounded response.
        Returns: (answer_text, latency_ms, model_used)
        """
        start_time = time.perf_counter()
        user_prompt = format_generation_prompt(query, retrieved_sources, detected_lang)

        # 1. Check if we have an active Gemini API key
        if self._client_initialized:
            import google.generativeai as genai

            # Try primary model first (gemini-2.5-flash / gemini-2.5-pro)
            import asyncio
            candidates = []
            for m in [self.model_name, self.fallback_model, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                if m and m not in candidates:
                    candidates.append(m)

            for model_candidate in candidates:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_candidate,
                        system_instruction=STRICT_RAG_SYSTEM_PROMPT,
                        generation_config={
                            "temperature": settings.TEMPERATURE,
                            "max_output_tokens": settings.MAX_GENERATION_TOKENS,
                        }
                    )
                    # Thread-safe synchronous call wrapped in asyncio.to_thread
                    response = await asyncio.to_thread(model.generate_content, user_prompt)
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    
                    if response and response.text:
                        return response.text.strip(), latency_ms, model_candidate
                except Exception as e:
                    print(f"[WARN] Error with model {model_candidate}: {e}. Retrying fallback...")

        # 2. Heuristic extraction fallback when API key is not supplied or offline
        fallback_answer = self._extractive_grounded_fallback(query, retrieved_sources, detected_lang)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return fallback_answer, latency_ms, "grounded_extractive_fallback"

    def _extractive_grounded_fallback(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        lang: str
    ) -> str:
        """
        Deterministic, grounded fallback that extracts the most relevant verified
        factual sentences directly from retrieved sources without hallucination.
        """
        from backend.generation.extractive_generator import DatasetAnswerExtractor
        extractor = DatasetAnswerExtractor()
        ans, _, _ = extractor.extract_answer(query, sources, detected_lang=lang, max_sentences=2)
        return ans
