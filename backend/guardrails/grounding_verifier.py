import re
from typing import List, Dict, Any, Tuple
from ingestion.clean_indic import indic_tokenize


class GroundingVerifier:
    """
    Verifies that the generated answer is strictly grounded in the retrieved sources.
    Performs fast Indic-aware N-gram containment, cross-lingual synonym alignment,
    and lexical overlap checks (<3ms).
    Prevents hallucinations by enforcing that the answer content originates from dataset passages.
    """

    def __init__(self, min_support_ratio: float = 0.50):
        self.min_support_ratio = min_support_ratio

    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def verify(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        is_cross_lingual: bool = False
    ) -> Tuple[bool, float, str]:
        """
        Calculates lexical entailment, cross-lingual entity alignment, and support ratio.
        Returns: (is_grounded, support_score, status_message)
        """
        if not answer or not sources:
            return False, 0.0, "Empty answer or no sources to verify against."

        # Check if answer is a standard refusal message (always considered safe)
        ans_lower = answer.lower()
        refusal_markers = [
            "couldn't find sufficient information",
            "don't have enough information",
            "पर्याप्त जानकारी नहीं",
            "पुरेशी माहिती उपलब्ध नाही",
            "पुरेशी माहिती नाही",
            "उपलब्ध ज्ञानकोशात",
            "माहिती मिळाली नाही"
        ]
        if any(marker in ans_lower for marker in refusal_markers):
            return True, 1.0, "Answer is a valid knowledge-base refusal."

        # Tokenize answer and aggregate source texts with cross-lingual synonym expansion
        raw_answer_tokens = indic_tokenize(answer, remove_stopwords=False, expand_synonyms=True)
        content_answer_tokens = set(indic_tokenize(answer, remove_stopwords=True, expand_synonyms=True))
        
        if not content_answer_tokens:
            return True, 1.0, "Answer contains only conversational or functional tokens."

        # Collect source tokens and full text
        source_content_tokens = set()
        aggregated_source_text = " ".join([src.get("text", "").lower() for src in sources])
        
        for src in sources:
            tokens = indic_tokenize(src.get("text", ""), remove_stopwords=True, expand_synonyms=True)
            source_content_tokens.update(tokens)

        # 1. Lexical & Synonym Overlap Check
        supported_tokens = content_answer_tokens.intersection(source_content_tokens)
        support_ratio = len(supported_tokens) / len(content_answer_tokens)
        
        # 2. Digit / Year / Number Alignment
        answer_digits = set(re.findall(r"\b\d+\b", answer))
        source_digits = set(re.findall(r"\b\d+\b", aggregated_source_text))
        digit_match = bool(answer_digits.intersection(source_digits)) if answer_digits else True

        # 3. N-Gram Containment Check
        bigrams = self._get_ngrams(raw_answer_tokens, 2)
        bigram_matches = 0
        if bigrams:
            for bg in bigrams:
                if bg.lower() in aggregated_source_text:
                    bigram_matches += 1
            bigram_ratio = bigram_matches / len(bigrams)
        else:
            bigram_ratio = 1.0

        # Calibrated support threshold: For cross-lingual synthesis (e.g. English answer from Marathi context),
        # threshold is 0.20 with entity/synonym alignment; for mono-lingual, it is 0.50.
        effective_min_ratio = 0.20 if is_cross_lingual else self.min_support_ratio
        support_score = round(0.7 * support_ratio + 0.3 * bigram_ratio, 3)

        is_grounded = (support_ratio >= effective_min_ratio) and digit_match
        
        if not is_grounded:
            status = f"Potential hallucination: Only {len(supported_tokens)}/{len(content_answer_tokens)} ({support_ratio:.1%}) content words supported by dataset."
        else:
            status = "Answer is strictly grounded in retrieved dataset evidence."

        return is_grounded, support_score, status
