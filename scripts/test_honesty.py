import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.orchestration.rag_orchestrator import RAGOrchestrator

async def main():
    orch = RAGOrchestrator()
    q = "ईमानदारी या सच्चाई की परिभाषा क्या है?"
    res = await orch.execute_rag(q)
    print("=== ORCHESTRATOR RESULT ===")
    print("Answer:", res.get("answer"))
    print("Model:", res.get("modelUsed"))
    print("Confidence:", res.get("confidence"))
    print("Refused:", res.get("refused"))
    print("Refusal Reason:", res.get("refusalReason"))
    print("\n--- SOURCES RETRIEVED (Top 3) ---")
    for i, s in enumerate(res.get("sources", [])):
        print(f"[{i+1}] (Score: {s.get('score')}) DocID: {s.get('docId')}")
        print(f"Text: {s.get('text')}\n")

asyncio.run(main())
