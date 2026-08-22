from typing import List, Dict, Any, Tuple
from backend.config import settings


class ConfidenceGuardrail:
    """
    Evaluates retrieval confidence from vector, BM25, and reranker scores.
    Rejects out-of-domain, irrelevant, and low-confidence queries before generation
    to ensure the model only answers questions backed by dataset evidence.
    """

    def __init__(self, threshold: float = settings.RETRIEVAL_CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def evaluate_confidence(self, sources: List[Dict[str, Any]], query: str = "") -> Tuple[bool, float, str]:
        """
        Evaluates top sources.
        Returns: (is_confident, confidence_score, explanation)
        """
        # Allow assistant capability / identity questions through
        q_lower = query.lower().strip()
        system_keywords = [
            "who are you", "what can you do", "what is vaani", "कोण आहेस", "काय करू शकतोस",
            "तुम कौन हो", "क्या कर सकते हो", "languages supported", "कोणत्या भाषा"
        ]
        if any(k in q_lower for k in system_keywords):
            return True, 0.95, "System capability and identity query."

        if not sources:
            return False, 0.0, "No relevant candidate passages found in the knowledge base."

        # Compute max and top-3 mean score
        scores = [float(s.get("score", 0.0)) for s in sources]
        max_score = max(scores)
        top3_mean = sum(scores[:3]) / len(scores[:3])

        # Combined confidence metric
        confidence = round(0.65 * max_score + 0.35 * top3_mean, 3)

        # Hard Gate 1: Absolute low relevance check (Out of domain)
        if max_score < 0.25:
            return (
                False,
                confidence,
                f"Query appears outside knowledge base domain (Max candidate score {max_score:.2f} < 0.25)."
            )

        # Hard Gate 2: Overall confidence threshold
        if confidence < self.threshold:
            return (
                False,
                confidence,
                f"Retrieval confidence ({confidence:.2f}) is below required confidence threshold ({self.threshold:.2f})."
            )

        return True, confidence, "Sufficient retrieval confidence grounded in knowledge base."
