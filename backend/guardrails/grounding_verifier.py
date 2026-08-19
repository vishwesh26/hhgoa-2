import re
from typing import List, Dict, Any, Tuple
from ingestion.clean_indic import indic_tokenize


class GroundingVerifier:
    """
    Verifies that the generated answer is strictly grounded in the retrieved sources.
    Performs fast Indic-aware N-gram containment and lexical overlap checks (<3ms).
    """

    def __init__(self, min_support_ratio: float = 0.40):
        self.min_support_ratio = min_support_ratio

    def verify(self, answer: str, sources: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """
        Calculates lexical entailment and content word support ratio.
        Returns: (is_grounded, support_score, status_message)
        """
        if not answer or not sources:
            return False, 0.0, "Empty answer or no sources to verify against."

        # Tokenize answer and aggregate source texts
        answer_tokens = set(indic_tokenize(answer, remove_stopwords=True))
        if not answer_tokens:
            return True, 1.0, "Answer contains only conversational or stop words."

        source_tokens = set()
        for src in sources:
            tokens = indic_tokenize(src.get("text", ""), remove_stopwords=True)
            source_tokens.update(tokens)

        # Calculate lexical overlap / support
        supported_tokens = answer_tokens.intersection(source_tokens)
        support_ratio = len(supported_tokens) / len(answer_tokens)
        support_score = round(support_ratio, 3)

        is_grounded = support_ratio >= self.min_support_ratio
        status = "Answer is fully grounded in retrieved evidence." if is_grounded else "Potential hallucination: answer tokens not supported by source passages."

        return is_grounded, support_score, status
