from typing import List, Dict, Any, Tuple
from backend.config import settings


class ConfidenceGuardrail:
    """
    Evaluates retrieval confidence from vector and BM25 scores.
    Rejects out-of-domain and low-confidence queries before wasting LLM inference latency.
    """

    def __init__(self, threshold: float = settings.RETRIEVAL_CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def evaluate_confidence(self, sources: List[Dict[str, Any]], query: str = "") -> Tuple[bool, float, str]:
        """
        Evaluates top sources.
        Returns: (is_confident, confidence_score, explanation)
        """
        # Allow capability / identity questions through
        q_lower = query.lower()
        if any(k in q_lower for k in ["language", "languages", "भाषा", "कोणती भाषा", "speak", "capabilities", "what can you do", "who are you"]):
            return True, 0.95, "System capability query."

        if not sources:
            return False, 0.0, "No relevant candidate passages found in knowledge base."

        # Compute max and top-3 mean score
        scores = [s.get("score", 0.0) for s in sources]
        max_score = max(scores)
        top3_mean = sum(scores[:3]) / len(scores[:3])

        # Combined confidence metric
        confidence = round(0.6 * max_score + 0.4 * top3_mean, 3)

        if confidence < self.threshold:
            return (
                False,
                confidence,
                f"Retrieval confidence ({confidence:.2f}) is below threshold ({self.threshold:.2f})."
            )

        return True, confidence, "Sufficient retrieval confidence."
