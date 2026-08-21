import sys
import os
import time
import json
import pyarrow.parquet as pq
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.orchestration.rag_orchestrator import RAGOrchestrator
from backend.retrieval.bm25_search import IndicBM25Searcher
from backend.retrieval.vector_search import QdrantVectorSearcher
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker

def evaluate_benchmark(val_parquet_path: str = "hinval.parquet", num_queries: int = 50):
    print("=" * 80)
    print(f"BENCHMARK EVALUATION ON {val_parquet_path} (Sample Size: {num_queries} queries)")
    print("=" * 80)

    pf = pq.ParquetFile(val_parquet_path)
    eval_queries = []
    
    for batch in pf.iter_batches(batch_size=100, columns=['query_id', 'query', 'Eng_Query', 'Answer', 'passages', 'query_type']):
        for r in batch.to_pylist():
            q_hi = str(r.get('query', '')).strip()
            q_id = r.get('query_id')
            passages = r.get('passages', {})
            if not isinstance(passages, dict):
                continue
            is_sel = passages.get('is_selected', [])
            if 1 not in is_sel:
                continue # Only evaluate queries with at least one ground-truth selected passage
                
            sel_indices = [idx for idx, s in enumerate(is_sel) if s == 1]
            ground_truth_doc_ids = [f"msmarco_hi_{q_id}_p{idx}" for idx in sel_indices]
            
            eval_queries.append({
                "query_id": q_id,
                "query": q_hi,
                "selected_indices": sel_indices,
                "ground_truth_doc_ids": ground_truth_doc_ids
            })
            if len(eval_queries) >= num_queries:
                break
        if len(eval_queries) >= num_queries:
            break

    print(f"Loaded {len(eval_queries)} ground-truth validation queries.\n")

    # Evaluators
    bm25 = IndicBM25Searcher("passage")
    vec = QdrantVectorSearcher("msmarco_xi_v2_passage")
    reranker = CrossEncoderReranker(top_k=5)

    metrics = {
        "BM25": {"r1": 0, "r5": 0, "r10": 0, "rr10": 0.0, "latencies": []},
        "Dense": {"r1": 0, "r5": 0, "r10": 0, "rr10": 0.0, "latencies": []},
        "Hybrid": {"r1": 0, "r5": 0, "r10": 0, "rr10": 0.0, "latencies": []},
        "Hybrid+Reranker": {"r1": 0, "r5": 0, "r10": 0, "rr10": 0.0, "latencies": []}
    }

    for i, eq in enumerate(eval_queries):
        q_text = eq["query"]
        target_docs = set(eq["ground_truth_doc_ids"])

        # 1. BM25
        t0 = time.perf_counter()
        bm25_res = bm25.search(q_text, top_k=20)
        lat_bm25 = (time.perf_counter() - t0) * 1000.0
        metrics["BM25"]["latencies"].append(lat_bm25)
        _score_run(bm25_res, target_docs, metrics["BM25"])

        # 2. Dense
        t0 = time.perf_counter()
        vec_res = vec.search(q_text, top_k=20)
        lat_vec = (time.perf_counter() - t0) * 1000.0
        metrics["Dense"]["latencies"].append(lat_vec)
        _score_run(vec_res, target_docs, metrics["Dense"])

        # 3. Hybrid RRF
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(vec_res, bm25_res, 0.5, 0.5, 60, 20)
        lat_hyb = (time.perf_counter() - t0) * 1000.0 + max(lat_bm25, lat_vec)
        metrics["Hybrid"]["latencies"].append(lat_hyb)
        _score_run(fused, target_docs, metrics["Hybrid"])

        # 4. Hybrid + Reranker
        t0 = time.perf_counter()
        reranked = reranker.rerank(q_text, fused[:15])
        lat_rr = (time.perf_counter() - t0) * 1000.0 + lat_hyb
        metrics["Hybrid+Reranker"]["latencies"].append(lat_rr)
        _score_run(reranked, target_docs, metrics["Hybrid+Reranker"])

    # Compute final aggregate scores
    N = len(eval_queries)
    print("=" * 80)
    print(f"{'RETRIEVAL METHOD':<20} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR@10':<10} | {'P50 (ms)':<10} | {'P90 (ms)':<10}")
    print("-" * 80)

    for method, m in metrics.items():
        r1 = (m["r1"] / N) * 100.0
        r5 = (m["r5"] / N) * 100.0
        r10 = (m["r10"] / N) * 100.0
        mrr = (m["rr10"] / N)
        lats = sorted(m["latencies"])
        p50 = lats[int(len(lats) * 0.50)] if lats else 0
        p90 = lats[int(len(lats) * 0.90)] if lats else 0

        print(f"{method:<20} | {r1:>8.1f}% | {r5:>8.1f}% | {r10:>8.1f}% | {mrr:>10.3f} | {p50:>8.1f}ms | {p90:>8.1f}ms")
    print("=" * 80)

def _score_run(results: List[Dict[str, Any]], target_docs: set, metric_dict: Dict[str, Any]):
    found_ranks = []
    for rank, r in enumerate(results, start=1):
        doc_id = r.get("doc_id", "")
        # Also check chunk_id prefix
        if doc_id in target_docs or any(t in r.get("chunk_id", "") for t in target_docs):
            found_ranks.append(rank)

    if found_ranks:
        best_rank = found_ranks[0]
        if best_rank <= 1:
            metric_dict["r1"] += 1
        if best_rank <= 5:
            metric_dict["r5"] += 1
        if best_rank <= 10:
            metric_dict["r10"] += 1
            metric_dict["rr10"] += (1.0 / best_rank)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    evaluate_benchmark(num_queries=n)
