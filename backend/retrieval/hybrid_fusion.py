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
        v_score = float(doc.get("score", doc.get("vector_score", 0.5)))
        if chunk_id not in doc_map:
            doc_map[chunk_id] = {**doc, "vector_rank": rank, "bm25_rank": None, "vector_score": v_score}
        else:
            doc_map[chunk_id]["vector_rank"] = rank
            doc_map[chunk_id]["vector_score"] = v_score

        # Score increment weighted by rank and relevance confidence
        score_increment = vector_weight * (1.0 / (rrf_k + rank)) * (0.5 + 0.5 * min(1.0, max(0.0, v_score)))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + score_increment

    # Process BM25 Results
    for rank, doc in enumerate(bm25_results, start=1):
        chunk_id = doc["chunk_id"]
        b_score = float(doc.get("score", doc.get("bm25_score", 0.5)))
        if chunk_id not in doc_map:
            doc_map[chunk_id] = {**doc, "vector_rank": None, "bm25_rank": rank, "bm25_score": b_score}
        else:
            doc_map[chunk_id]["bm25_rank"] = rank
            doc_map[chunk_id]["bm25_score"] = b_score

        # Score increment weighted by rank and relevance confidence
        score_increment = bm25_weight * (1.0 / (rrf_k + rank)) * (0.5 + 0.5 * min(1.0, max(0.0, b_score)))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + score_increment

    # Rank all candidate chunk IDs by combined RRF score
    ranked_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    # Ensure top-3 BM25 and top-3 Vector candidates are prioritized in the candidate pool for reranker
    guaranteed_cids = []
    for doc in bm25_results[:3]:
        cid = doc.get("chunk_id")
        if cid and cid not in guaranteed_cids:
            guaranteed_cids.append(cid)
    for doc in vector_results[:3]:
        cid = doc.get("chunk_id")
        if cid and cid not in guaranteed_cids:
            guaranteed_cids.append(cid)

    # Merge guaranteed items first, followed by the rest of ranked items
    final_ordered_cids = list(guaranteed_cids)
    for cid in ranked_chunk_ids:
        if cid not in final_ordered_cids:
            final_ordered_cids.append(cid)

    max_possible_rrf = (vector_weight + bm25_weight) * (1.0 / (rrf_k + 1))

    fused_results = []
    for cid in final_ordered_cids[:top_k]:
        if cid not in doc_map:
            continue
        item = doc_map[cid]
        raw_rrf = scores.get(cid, 0.0)
        normalized_rrf = min(1.0, raw_rrf / max_possible_rrf) if max_possible_rrf > 0 else 0.5
        item["rrf_score"] = raw_rrf
        item["score"] = normalized_rrf
        fused_results.append(item)

    return fused_results
