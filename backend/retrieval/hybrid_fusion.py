from typing import List, Dict, Any
from backend.config import settings


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    vector_weight: float = settings.DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = settings.DEFAULT_BM25_WEIGHT,
    rrf_k: int = settings.RRF_K,
    top_k: int = settings.VECTOR_TOP_K
) -> List[Dict[str, Any]]:
    """
    Combines dense vector search results and lexical BM25 results using
    weighted Reciprocal Rank Fusion (RRF):
    RRF_Score(doc) = w_vector * (1 / (k + rank_vector)) + w_bm25 * (1 / (k + rank_bm25))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Process Vector Results
    for rank, doc in enumerate(vector_results, start=1):
        chunk_id = doc["chunk_id"]
        if chunk_id not in doc_map:
            doc_map[chunk_id] = {**doc, "vector_rank": rank, "bm25_rank": None}
        else:
            doc_map[chunk_id]["vector_rank"] = rank
            doc_map[chunk_id]["vector_score"] = doc.get("vector_score", 0.0)

        score_increment = vector_weight * (1.0 / (rrf_k + rank))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + score_increment

    # Process BM25 Results
    for rank, doc in enumerate(bm25_results, start=1):
        chunk_id = doc["chunk_id"]
        if chunk_id not in doc_map:
            doc_map[chunk_id] = {**doc, "vector_rank": None, "bm25_rank": rank}
        else:
            doc_map[chunk_id]["bm25_rank"] = rank
            doc_map[chunk_id]["bm25_score"] = doc.get("bm25_score", 0.0)

        score_increment = bm25_weight * (1.0 / (rrf_k + rank))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + score_increment

    # Rank by combined RRF score
    ranked_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    max_possible_rrf = (vector_weight + bm25_weight) * (1.0 / (rrf_k + 1))

    fused_results = []
    for cid in ranked_chunk_ids[:top_k]:
        item = doc_map[cid]
        raw_rrf = scores[cid]
        normalized_rrf = min(1.0, raw_rrf / max_possible_rrf)
        item["rrf_score"] = raw_rrf
        item["score"] = normalized_rrf
        fused_results.append(item)

    return fused_results
