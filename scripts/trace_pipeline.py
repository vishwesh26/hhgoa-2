import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.orchestration.rag_orchestrator import RAGOrchestrator
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion

async def inspect():
    orch = RAGOrchestrator()
    q = "ईमानदारी या सच्चाई की परिभाषा"
    
    vec_searcher = orch.vector_searchers["sentence"]
    bm25_searcher = orch.bm25_searchers["sentence"]
    
    vec_results = vec_searcher.search(q, 20)
    bm25_results = bm25_searcher.search(q, 20)
    
    print(f"BM25 Results ({len(bm25_results)}):")
    for r in bm25_results[:5]:
        print(f"  Doc: {r.get('doc_id')} | Chunk: {r.get('chunk_id')} | Score: {r.get('score'):.3f}")
        
    fused = reciprocal_rank_fusion(vec_results, bm25_results, 0.5, 0.5, 60, 20)
    print(f"\nFused Results ({len(fused)}):")
    for r in fused[:5]:
        print(f"  Doc: {r.get('doc_id')} | Chunk: {r.get('chunk_id')} | Score: {r.get('score'):.3f}")
        
    reranked = orch.reranker.rerank(q, fused)
    print(f"\nReranked Results ({len(reranked)}):")
    for r in reranked[:5]:
        print(f"  Doc: {r.get('doc_id')} | Chunk: {r.get('chunk_id')} | Score: {r.get('score'):.3f}")
        print(f"  Text: {r.get('text')[:120]}\n")

asyncio.run(inspect())
