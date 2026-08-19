import time
from typing import Dict, Any, List
from contextlib import contextmanager


class LatencyTracker:
    """
    High-precision per-stage telemetry tracker for end-to-end RAG pipelines.
    """

    def __init__(self):
        self.stages: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.global_start = time.perf_counter()

    @contextmanager
    def measure(self, stage_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.stages[stage_name] = round((t1 - t0) * 1000.0, 2)

    def record_stage(self, stage_name: str, duration_ms: float):
        self.stages[stage_name] = round(duration_ms, 2)

    def get_summary(self) -> Dict[str, Any]:
        total_time_ms = round((time.perf_counter() - self.global_start) * 1000.0, 2)
        # Calculate RAG-specific pipeline latency (excluding STT audio upload)
        rag_stages = ["queryAnalysis", "embedding", "vectorSearch", "bm25Search", "fusion", "reranking", "generation", "grounding"]
        rag_latency = round(sum(self.stages.get(k, 0.0) for k in rag_stages), 2)

        return {
            "stages": self.stages,
            "ragLatencyMs": rag_latency,
            "totalLatencyMs": total_time_ms
        }
