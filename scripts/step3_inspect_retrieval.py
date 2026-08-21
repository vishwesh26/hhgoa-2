import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.retrieval.bm25_search import IndicBM25Searcher
from backend.retrieval.vector_search import QdrantVectorSearcher
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker

def inspect_retrieval(query):
    print("=" * 80)
    print(f"3. RETRIEVAL INSPECTION FOR QUERY: '{query}'")
    print("=" * 80)

    # 1. BM25 Top 10
    bm25 = IndicBM25Searcher("combined")
    bm25_results = bm25.search(query, top_k=10)
    print("\n--- TOP 10 BM25 RESULTS ---")
    for i, r in enumerate(bm25_results):
        print(f"[{i+1}] Score: {r['score']:.4f} | Chunk ID: {r['chunk_id']} | Doc ID: {r.get('doc_id')}")
        print(f"    Passage: {r['text'][:140]}...\n")

    # 2. Vector Top 10
    vec = QdrantVectorSearcher("vaani_msmarco_sentence")
    vec_results = vec.search(query, top_k=10)
    print("\n--- TOP 10 VECTOR RESULTS ---")
    for i, r in enumerate(vec_results):
        print(f"[{i+1}] Cosine/Distance Score: {r['score']:.4f} | Chunk ID: {r['chunk_id']} | Doc ID: {r.get('doc_id')}")
        print(f"    Passage: {r['text'][:140]}...\n")

    # 3. Hybrid RRF Top 10
    fused = reciprocal_rank_fusion(
        vector_results=vec_results,
        bm25_results=bm25_results,
        vector_weight=0.5,
        bm25_weight=0.5,
        rrf_k=60,
        top_k=10
    )
    print("\n--- TOP 10 HYBRID RESULTS ---")
    for i, r in enumerate(fused):
        v_rank = r.get("vector_rank")
        b_rank = r.get("bm25_rank")
        print(f"[{i+1}] Final RRF Score: {r['score']:.4f} (Vector Rank: {v_rank}, BM25 Rank: {b_rank}) | Chunk ID: {r['chunk_id']}")
        print(f"    Passage: {r['text'][:140]}...\n")

    # 4. Cross-Encoder Reranked Top 5
    reranker = CrossEncoderReranker(top_k=5)
    reranked = reranker.rerank(query, fused)
    print("\n--- TOP 5 RERANKED RESULTS ---")
    for i, r in enumerate(reranked):
        print(f"[{i+1}] Blended Score: {r['score']:.4f} (Raw Rerank Score: {r.get('rerank_score'):.4f}) | Chunk ID: {r['chunk_id']}")
        print(f"    Passage: {r['text'][:140]}...\n")

inspect_retrieval("कॉर्पोरेशन मतलब क्या?")
