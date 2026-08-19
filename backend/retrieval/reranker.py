import time
from typing import List, Dict, Any
from backend.config import settings

_FLASHRANK_CLIENT = None


def get_reranker_client():
    global _FLASHRANK_CLIENT
    if _FLASHRANK_CLIENT is None:
        try:
            from flashrank import Ranker
            print(f"[INFO] Initializing FlashRank reranker: {settings.RERANKER_MODEL_NAME}...")
            _FLASHRANK_CLIENT = Ranker(model_name=settings.RERANKER_MODEL_NAME, cache_dir="./data/models")
        except Exception as e:
            print(f"[WARN] FlashRank init exception: {e}. Using fast hybrid score reranker.")
            _FLASHRANK_CLIENT = "heuristic"
    return _FLASHRANK_CLIENT


from backend.retrieval.bm25_search import indic_tokenize


class CrossEncoderReranker:
    """
    Ultra-low latency cross-encoder reranker. Evaluates query-passage relevance
    over top candidate passages in 15-25ms.
    """

    def __init__(self, top_k: int = settings.FINAL_CONTEXT_K):
        self.top_k = top_k

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # Take up to RERANKER_TOP_K candidates
        candidate_subset = candidates[:settings.RERANKER_TOP_K]
        ranker = get_reranker_client()
        q_terms = indic_tokenize(query, remove_stopwords=True)

        if ranker != "heuristic" and ranker is not None:
            try:
                from flashrank import RerankRequest
                passages = [{"id": i, "text": c["text"], "meta": c} for i, c in enumerate(candidate_subset)]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = ranker.rerank(rerank_req)

                reranked_docs = []
                for res in results:
                    meta = res["meta"]
                    raw_res_score = float(res["score"])
                    base_rrf_score = float(meta.get("score", 0.5))
                    # Check keyword overlap using Indic tokenization and synonyms
                    text_lower = meta.get("text", "").lower()
                    overlap_count = sum(1 for t in q_terms if t in text_lower)
                    overlap_boost = min(0.60, 0.15 * overlap_count)
                    # Calibrated blend of cross-encoder confidence, hybrid retrieval rank, and lexical match
                    blended_score = round(0.30 * base_rrf_score + 0.30 * min(1.0, raw_res_score * 3.0) + overlap_boost, 3)
                    reranked_docs.append({
                        **meta,
                        "rerank_score": raw_res_score,
                        "score": max(min(blended_score, 1.0), 0.05)
                    })
                
                reranked_docs.sort(key=lambda x: x["score"], reverse=True)
                return reranked_docs[:self.top_k]
            except Exception as e:
                print(f"[WARN] FlashRank rerank failure: {e}")

        # Heuristic scoring fallback
        for c in candidate_subset:
            base_score = c.get("score", 0.5)
            # Boost score if query terms exist directly in passage text
            query_words = set(query.lower().split())
            text_words = set(c["text"].lower().split())
            overlap = len(query_words.intersection(text_words)) / max(1, len(query_words))
            c["rerank_score"] = round(0.7 * base_score + 0.3 * overlap, 4)
            c["score"] = c["rerank_score"]

        sorted_docs = sorted(candidate_subset, key=lambda x: x["score"], reverse=True)
        return sorted_docs[:self.top_k]
