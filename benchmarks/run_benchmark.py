import json
import asyncio
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from backend.orchestration.rag_orchestrator import RAGOrchestrator


BENCHMARK_DATASET_PATH = Path("./benchmarks/dataset/indic_rag_bench_300.json")
RESULTS_DIR = Path("./benchmarks/results")
REPORT_PATH = RESULTS_DIR / "benchmark_report.json"


async def run_benchmarks_async(orchestrator: RAGOrchestrator, sample_size: int = 50) -> Dict[str, Any]:
    """
    Executes benchmark evaluations across the Indic RAG dataset and computes real metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not BENCHMARK_DATASET_PATH.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {BENCHMARK_DATASET_PATH}")

    with open(BENCHMARK_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Repeat or slice to reach sample_size
    test_queries = []
    while len(test_queries) < sample_size:
        test_queries.extend(dataset)
    test_queries = test_queries[:sample_size]

    print(f"[INFO] Running benchmark on {len(test_queries)} queries...")

    rag_latencies = []
    total_latencies = []
    stage_latencies: Dict[str, List[float]] = {
        "queryAnalysis": [],
        "vectorSearch": [],
        "bm25Search": [],
        "fusion": [],
        "reranking": [],
        "generation": [],
        "grounding": []
    }

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    grounded_count = 0
    refusal_correct_count = 0
    total_refusal_expected = 0

    for idx, item in enumerate(test_queries, start=1):
        query = item["query"]
        expected_doc = item.get("expected_doc_id")
        should_refuse = item.get("should_refuse", False)

        t0 = time.perf_counter()
        result = await orchestrator.execute_rag(query)
        t_total = (time.perf_counter() - t0) * 1000.0

        lat_info = result.get("latency", {})
        rag_lat = lat_info.get("ragLatencyMs", t_total)
        rag_latencies.append(rag_lat)
        total_latencies.append(t_total)

        for stage_name in stage_latencies.keys():
            st_val = lat_info.get("stages", {}).get(stage_name, 0.0)
            stage_latencies[stage_name].append(st_val)

        # Evaluate Grounding
        if result.get("grounded", False):
            grounded_count += 1

        # Evaluate Refusal Guardrail
        if should_refuse:
            total_refusal_expected += 1
            if result.get("refused", False):
                refusal_correct_count += 1

        # Evaluate Retrieval Metrics (Recall & MRR)
        if expected_doc:
            sources = result.get("sources", [])
            retrieved_doc_ids = [s.get("docId", "") for s in sources]

            if retrieved_doc_ids and expected_doc in retrieved_doc_ids[:1]:
                hits_at_1 += 1
            if retrieved_doc_ids and expected_doc in retrieved_doc_ids[:3]:
                hits_at_3 += 1
            if retrieved_doc_ids and expected_doc in retrieved_doc_ids[:5]:
                hits_at_5 += 1

            if expected_doc in retrieved_doc_ids:
                rank = retrieved_doc_ids.index(expected_doc) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

    # Compute Statistical Metrics
    n_queries = len(test_queries)
    n_eval_docs = max(1, len(reciprocal_ranks))

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries_tested": n_queries,
        "latency_metrics": {
            "rag_pipeline": {
                "p50_ms": round(float(np.percentile(rag_latencies, 50)), 2),
                "p70_ms": round(float(np.percentile(rag_latencies, 70)), 2),
                "p90_ms": round(float(np.percentile(rag_latencies, 90)), 2),
                "p100_ms": round(float(np.max(rag_latencies)), 2),
                "mean_ms": round(float(np.mean(rag_latencies)), 2),
                "min_ms": round(float(np.min(rag_latencies)), 2),
                "max_ms": round(float(np.max(rag_latencies)), 2),
            },
            "total_system": {
                "p50_ms": round(float(np.percentile(total_latencies, 50)), 2),
                "p70_ms": round(float(np.percentile(total_latencies, 70)), 2),
                "p100_ms": round(float(np.max(total_latencies)), 2),
                "mean_ms": round(float(np.mean(total_latencies)), 2),
            },
            "stage_breakdown_p50_ms": {
                stage: round(float(np.percentile(vals, 50)), 2)
                for stage, vals in stage_latencies.items() if vals
            }
        },
        "retrieval_quality": {
            "recall_at_1_pct": round((hits_at_1 / n_eval_docs) * 100.0, 2),
            "recall_at_3_pct": round((hits_at_3 / n_eval_docs) * 100.0, 2),
            "recall_at_5_pct": round((hits_at_5 / n_eval_docs) * 100.0, 2),
            "mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 1.0, 3),
        },
        "guardrail_quality": {
            "grounded_answers_pct": round((grounded_count / n_queries) * 100.0, 2),
            "refusal_accuracy_pct": round((refusal_correct_count / max(1, total_refusal_expected)) * 100.0, 2)
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Benchmark complete. Report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    from backend.api.routes_rag import get_orchestrator
    orch = get_orchestrator()
    asyncio.run(run_benchmarks_async(orch, sample_size=30))
