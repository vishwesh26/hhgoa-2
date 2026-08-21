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
            # Try model fallback candidates in prioritized order
            # Try candidate models in prioritized fast order
            candidate_models = [
                self.model_name,
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-2.0-flash",
                "gemini-3.5-flash",
                self.fallback_model,
            ]
            seen_models = set()
            import asyncio
            for model_candidate in candidate_models:
                if not model_candidate or model_candidate in seen_models:
                    continue
                seen_models.add(model_candidate)
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
                    
                    answer_text = ""
                    if response:
                        try:
                            if hasattr(response, "text") and response.text:
                                answer_text = response.text.strip()
                        except Exception:
                            pass
                        if not answer_text and hasattr(response, "candidates") and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                                parts_text = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
                                if parts_text:
                                    answer_text = "".join(parts_text).strip()
                    
                    if answer_text:
                        return answer_text, latency_ms, model_candidate
                except Exception as e:
                    print(f"[WARN] Gemini model {model_candidate} error: {e}. Retrying next candidate...")

        # 2. Honest refusal fallback when LLM is unavailable
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        if detected_lang == "hi":
            fallback_answer = "दिए गए संदर्भ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।"
        elif detected_lang == "mr":
            fallback_answer = "दिलेल्या संदर्भात या प्रश्नाचे उत्तर देण्यासाठी पुरेशी माहिती उपलब्ध नाही."
        else:
            fallback_answer = "I don't have enough information in the provided knowledge base to answer that reliably."
        return fallback_answer, latency_ms, "honest_refusal_fallback"
