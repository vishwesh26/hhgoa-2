import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.orchestration.rag_orchestrator import RAGOrchestrator
from backend.generation.prompts import format_generation_prompt

async def debug_query(q):
    orch = RAGOrchestrator()
    print(f"=== DEBUGGING QUERY: {q} ===")
    
    # 1. Routing
    strategy = orch.router.route_query(q)
    print("Router Strategy:", strategy)
    
    # 2. Retrieval
    vec_searcher = orch.vector_searchers["sentence"]
    bm25_searcher = orch.bm25_searchers["sentence"]
    
    vec_results = await vec_searcher.search(q, top_k=10)
    bm25_results = await bm25_searcher.search(q, top_k=10)
    
    print(f"\n[Vector Results]: {len(vec_results)}")
    for i, r in enumerate(vec_results[:3]):
        print(f"  V{i+1}: ({r.get('score'):.3f}) {r.get('text', '')[:100]}...")
        
    print(f"\n[BM25 Results]: {len(bm25_results)}")
    for i, r in enumerate(bm25_results[:3]):
        print(f"  B{i+1}: ({r.get('score'):.3f}) {r.get('text', '')[:100]}...")
        
    # 3. Hybrid Fusion
    fused = orch.hybrid_fusion.fuse(vec_results, bm25_results)
    print(f"\n[Fused Results]: {len(fused)}")
    for i, r in enumerate(fused[:3]):
        print(f"  F{i+1}: ({r.get('score'):.3f}) {r.get('text', '')[:100]}...")
        
    # 4. Rerank
    reranked = orch.reranker.rerank(q, fused)
    print(f"\n[Reranked Results]: {len(reranked)}")
    for i, r in enumerate(reranked[:3]):
        print(f"  R{i+1}: ({r.get('score'):.3f}) {r.get('text', '')[:100]}...")
        
    # 5. Generation Prompt
    prompt = format_generation_prompt(q, reranked[:5], detected_lang="hi")
    print("\n[Formatted Prompt for Gemini]:")
    print(prompt)

asyncio.run(debug_query("ईमानदारी या सच्चाई की परिभाषा क्या है?"))
