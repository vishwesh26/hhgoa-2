import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.retrieval.bm25_search import IndicBM25Searcher
from backend.retrieval.vector_search import QdrantVectorSearcher

async def inspect():
    bm25 = IndicBM25Searcher("sentence")
    bm25_comb = IndicBM25Searcher("combined")
    
    q = "ईमानदारी या सच्चाई की परिभाषा"
    print(f"Query: {q}")
    
    # 1. Search BM25 sentence
    res_sent = bm25.search(q, top_k=5)
    print(f"\n--- BM25 Sentence Results ({len(res_sent)}) ---")
    for r in res_sent:
        print(f"Score: {r['score']:.3f} | Chunk: {r['chunk_id']} | Text: {r['text'][:120]}")
        
    # 2. Search BM25 combined
    res_comb = bm25_comb.search(q, top_k=5)
    print(f"\n--- BM25 Combined Results ({len(res_comb)}) ---")
    for r in res_comb:
        print(f"Score: {r['score']:.3f} | Chunk: {r['chunk_id']} | Text: {r['text'][:120]}")
        
    # 3. Vector Search
    vec = QdrantVectorSearcher("vaani_msmarco_sentence")
    vec_res = vec.search(q, top_k=5)
    print(f"\n--- Qdrant Vector Sentence Results ({len(vec_res)}) ---")
    for r in vec_res:
        print(f"Score: {r['score']:.3f} | Chunk: {r['chunk_id']} | Text: {r['text'][:120]}")

asyncio.run(inspect())
