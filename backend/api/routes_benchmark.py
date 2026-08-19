import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, Optional
from backend.api.routes_rag import get_orchestrator
from backend.orchestration.rag_orchestrator import RAGOrchestrator

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark"])

BENCHMARK_REPORT_PATH = Path("./benchmarks/results/benchmark_report.json")


@router.get("/results")
async def get_benchmark_results() -> Dict[str, Any]:
    """
    Returns the latest benchmark report containing real P50, P70, P100 latencies and quality metrics.
    """
    if not BENCHMARK_REPORT_PATH.exists():
        return {
            "status": "not_run",
            "message": "No benchmark report found. Run POST /api/benchmark/run to trigger benchmarking."
        }

    try:
        with open(BENCHMARK_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read benchmark report: {str(e)}")


@router.post("/run")
async def trigger_benchmark(
    background_tasks: BackgroundTasks,
    sample_size: int = 50,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Triggers an asynchronous benchmark run on the Indic RAG dataset.
    """
    from benchmarks.run_benchmark import run_benchmarks_async
    background_tasks.add_task(run_benchmarks_async, orchestrator, sample_size)
    return {
        "status": "started",
        "message": f"Benchmark triggered with sample size {sample_size}. Check /api/benchmark/results shortly."
    }
