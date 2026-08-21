import sys
import time
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.orchestration.rag_orchestrator import RAGOrchestrator


async def run_benchmark():
    orchestrator = RAGOrchestrator()
    queries = [
        "मॅनहॅटन प्रकल्पाच्या यशाचा तात्काळ काय परिणाम झाला?",
        "पुनर्संचयी न्यायाने पीडितांचे काय समाधान होते?",
        "What is photosynthesis?",
        "प्रकाश संश्लेषण क्या है?",
        "Photosynthesis kaise work karta hai?",
        "सूर्यापासून पृथ्वीपर्यंत प्रकाश पोहोचायला किती वेळ लागतो?",
        "What is the chemical equation for photosynthesis?",
        "मॅनहॅटन प्रकल्प म्हणजे काय?"
    ]

    print("=" * 110)
    print(f"{'QUERY':<45} | {'LANG':<4} | {'VEC':<6} | {'BM25':<6} | {'RERANK':<7} | {'TOTAL':<8} | {'STATUS'}")
    print("=" * 110)

    # Warmup query
    await orchestrator.execute_rag("warmup query")

    all_latencies = []

    for q in queries:
        start = time.perf_counter()
        resp = await orchestrator.execute_rag(q)
        total_time = (time.perf_counter() - start) * 1000.0
        all_latencies.append(total_time)

        lat = resp.get("latency", {})
        vec_ms = lat.get("vectorSearchMs", 0)
        bm25_ms = lat.get("bm25SearchMs", 0)
        rerank_ms = lat.get("rerankMs", 0)
        lang = resp.get("metadata", {}).get("detected_language", "en")
        badge = "⚡ UNDER 200MS" if total_time < 200 else "⚠️ EXCEEDS 200MS"

        print(f"{q[:44]:<45} | {lang:<4} | {vec_ms:5.1f} | {bm25_ms:5.1f} | {rerank_ms:6.1f} | {total_time:6.1f} ms | [{badge}]")

    print("=" * 110)
    avg_latency = sum(all_latencies) / len(all_latencies)
    max_latency = max(all_latencies)
    print(f"AVERAGE LATENCY: {avg_latency:.1f} ms | MAX LATENCY: {max_latency:.1f} ms | P95: {sorted(all_latencies)[int(len(all_latencies)*0.95)]:.1f} ms")
    print("=" * 110)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
