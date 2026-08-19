import pytest
import asyncio
from backend.orchestration.query_analyzer import QueryAnalyzer
from backend.orchestration.rag_orchestrator import RAGOrchestrator


def test_query_analyzer_codemix():
    analyzer = QueryAnalyzer()
    
    # Marathi-English Code-Mixed
    res1 = analyzer.analyze("Photosynthesis म्हणजे exactly काय?")
    assert res1["language"] == "mr"
    assert res1["is_code_mixed"] is True

    # Hindi-English Code-Mixed
    res2 = analyzer.analyze("Photosynthesis ka process kaise work karta hai?")
    assert res2["language"] == "hi"
    assert res2["is_code_mixed"] is True

    # Factual Query Type
    res3 = analyzer.analyze("When did India become independent?")
    assert res3["query_type"] == "factual"
    assert res3["adaptive_strategy"]["target_collection"] == "sentence"


@pytest.mark.asyncio
async def test_orchestrator_execution():
    orchestrator = RAGOrchestrator()
    response = await orchestrator.execute_rag("What is the chemical equation for photosynthesis?")
    assert "answer" in response
    assert "latency" in response
    assert response["latency"]["totalLatencyMs"] > 0
    assert response["confidence"] >= 0.0
