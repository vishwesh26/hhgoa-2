import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.orchestration.rag_orchestrator import RAGOrchestrator
from backend.retrieval.bm25_search import IndicBM25Searcher
from backend.retrieval.vector_search import QdrantVectorSearcher
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker

async def run_test(query_text: str):
    orch = RAGOrchestrator()
    analysis = orch.query_analyzer.analyze(query_text)
    lang = analysis["language"]
    strategy = analysis["adaptive_strategy"]
    target_col = strategy["target_collection"]

    print("QUERY")
    print(query_text)
    print()

    print("LANGUAGE")
    print("Hindi" if lang == "hi" else ("Marathi" if lang == "mr" else "English"))
    print()

    # Parallel retrieval
    vec_searcher = orch.vector_searchers.get(target_col, orch.vector_searchers["combined"])
    bm25_searcher = orch.bm25_searchers.get(target_col, orch.bm25_searchers["combined"])
    bm25_comb = orch.bm25_searchers.get("combined", orch.bm25_searchers["sentence"])

    vec_res = vec_searcher.search(query_text, top_k=10)
    bm25_res = bm25_searcher.search(query_text, top_k=10)
    bm25_comb_res = bm25_comb.search(query_text, top_k=10)

    # Merge BM25
    seen = set()
    merged_bm25 = []
    for r in (bm25_res or []) + (bm25_comb_res or []):
        cid = r.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            merged_bm25.append(r)

    print("TOP VECTOR RESULTS")
    if not vec_res:
        print("(No vector results returned)")
    for i, r in enumerate(vec_res[:5]):
        print(f"{i+1}. [Score: {r.get('score', 0):.4f}] [Chunk: {r.get('chunk_id')}]")
        print(f"   {r.get('text', '')[:140]}...")
    print()

    print("TOP BM25 RESULTS")
    if not merged_bm25:
        print("(No BM25 results returned)")
    for i, r in enumerate(merged_bm25[:5]):
        print(f"{i+1}. [Score: {r.get('score', 0):.4f}] [Chunk: {r.get('chunk_id')}]")
        print(f"   {r.get('text', '')[:140]}...")
    print()

    # Hybrid fusion
    fused = reciprocal_rank_fusion(
        vector_results=vec_res,
        bm25_results=merged_bm25,
        vector_weight=strategy.get("vector_weight", 0.5),
        bm25_weight=strategy.get("bm25_weight", 0.5),
        rrf_k=60,
        top_k=10
    )

    print("HYBRID RESULTS")
    if not fused:
        print("(No hybrid fused results)")
    for i, r in enumerate(fused[:5]):
        print(f"{i+1}. [RRF Score: {r.get('score', 0):.4f}] [VecRank: {r.get('vector_rank')}, BM25Rank: {r.get('bm25_rank')}] [Chunk: {r.get('chunk_id')}]")
        print(f"   {r.get('text', '')[:140]}...")
    print()

    # Reranking
    reranker = CrossEncoderReranker(top_k=5)
    reranked = reranker.rerank(query_text, fused)

    print("RERANKED RESULTS")
    if not reranked:
        print("(No reranked results)")
    for i, r in enumerate(reranked[:5]):
        print(f"{i+1}. [Blended Score: {r.get('score', 0):.4f}] [Raw Score: {r.get('rerank_score', 0):.4f}] [Chunk: {r.get('chunk_id')}]")
        print(f"   {r.get('text', '')[:140]}...")
    print()

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "कॉर्पोरेशन मतलब क्या?"
    asyncio.run(run_test(q))
