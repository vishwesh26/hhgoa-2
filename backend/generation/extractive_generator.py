import time
from typing import List, Dict, Any, Tuple
from backend.chunking.sentence_chunker import SentenceChunker
from backend.retrieval.bm25_search import indic_tokenize


class DatasetAnswerExtractor:
    """
    Ultra-low latency (<2ms), zero-token dataset-grounded answer extractor.
    Extracts the most relevant factual answer sentence(s) directly from the
    retrieved MSMARCO-XI dataset passages.
    """

    def __init__(self):
        self.sentence_chunker = SentenceChunker()

    def extract_answer(
        self,
        query: str,
        retrieved_sources: List[Dict[str, Any]],
        detected_lang: str = "en",
        max_sentences: int = 2
    ) -> Tuple[str, float, str]:
        """
        Extracts verified factual answer sentences from top retrieved sources.
        Returns: (answer_text, latency_ms, extractor_method)
        """
        start_time = time.perf_counter()

        if not retrieved_sources:
            if detected_lang == "hi":
                ans = "मुझे इस प्रश्न का उत्तर देने के लिए ज्ञानकोष में पर्याप्त जानकारी नहीं मिली।"
            elif detected_lang == "mr":
                ans = "या प्रश्नाचे उत्तर देण्यासाठी उपलब्ध ज्ञानकोशात पुरेशी माहिती उपलब्ध नाही."
            else:
                ans = "I couldn't find sufficient information in the provided knowledge base to answer that reliably."
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ans, latency_ms, "extractive_refusal"

        # Tokenize query for scoring
        query_tokens = set(indic_tokenize(query, remove_stopwords=True, lang=detected_lang))
        if not query_tokens:
            query_tokens = set(query.lower().split())

        scored_sentences = []

        for src_idx, src in enumerate(retrieved_sources[:3]):
            text = src.get("text", "")
            sentences = self.sentence_chunker.split_into_sentences(text)
            
            for s_idx, sent in enumerate(sentences):
                sent_clean = sent.strip()
                if len(sent_clean.split()) < 4:
                    continue

                sent_tokens = set(indic_tokenize(sent_clean, remove_stopwords=False, lang=src.get("language", detected_lang)))
                overlap = len(query_tokens.intersection(sent_tokens))
                
                # Bonus for exact key terms matching and top source position
                source_weight = 1.0 / (src_idx + 1)
                position_weight = 1.0 / (s_idx + 1)
                score = (overlap * 2.0) + (source_weight * 1.5) + (position_weight * 0.5)

                scored_sentences.append({
                    "sentence": sent_clean,
                    "score": score,
                    "source_id": src.get("chunk_id", ""),
                    "doc_id": src.get("doc_id", "")
                })

        if not scored_sentences:
            best_source = retrieved_sources[0].get("text", "")
            ans = best_source[:300] + "..." if len(best_source) > 300 else best_source
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ans, latency_ms, "dataset_passage_slice"

        # Sort by relevance score and deduplicate
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)
        top_sentences = []
        seen_texts = set()
        for s in scored_sentences:
            normalized = s["sentence"].lower().strip(" .?!।॥")
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                top_sentences.append(s["sentence"])
                if len(top_sentences) >= max_sentences:
                    break
        
        # Ensure punctuation
        joined_ans = " ".join(top_sentences)
        if not any(joined_ans.endswith(p) for p in [".", "।", "!", "?"]):
            joined_ans += "."

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return joined_ans, latency_ms, "dataset_extractive_grounding"
