import pytest
from backend.retrieval.bm25_search import IndicBM25Searcher
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker


def test_bm25_search():
    sample_chunks = [
        {"chunk_id": "c1", "text": "Photosynthesis is the process of converting sunlight to energy.", "language": "en"},
        {"chunk_id": "c2", "text": "The Ganga is the longest river in India.", "language": "en"},
        {"chunk_id": "c3", "text": "सूर्य का प्रकाश पृथ्वी तक 8 मिनट 20 सेकंड में पहुँचता है।", "language": "hi"}
    ]
    bm25 = IndicBM25Searcher("test")
    bm25.build_index(sample_chunks)

    results = bm25.search("Photosynthesis sunlight", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c1"


def test_reciprocal_rank_fusion():
    vec_results = [
        {"chunk_id": "c1", "score": 0.9, "text": "Doc 1"},
        {"chunk_id": "c2", "score": 0.8, "text": "Doc 2"}
    ]
    bm25_results = [
        {"chunk_id": "c2", "score": 0.95, "text": "Doc 2"},
        {"chunk_id": "c3", "score": 0.7, "text": "Doc 3"}
    ]

    fused = reciprocal_rank_fusion(vec_results, bm25_results)
    assert len(fused) == 3
    # c2 appeared in both, so it should rank first
    assert fused[0]["chunk_id"] == "c2"


def test_reranker():
    reranker = CrossEncoderReranker(top_k=2)
    candidates = [
        {"chunk_id": "c1", "score": 0.5, "text": "Photosynthesis is vital for plants."},
        {"chunk_id": "c2", "score": 0.6, "text": "Something about space and rockets."}
    ]
    reranked = reranker.rerank("photosynthesis plant process", candidates)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c1"
