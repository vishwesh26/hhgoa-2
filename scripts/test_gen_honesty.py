import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from backend.generation.llm_generator import GroundedLLMGenerator

async def test_gen():
    gen = GroundedLLMGenerator()
    q = "ईमानदारी या सच्चाई की परिभाषा क्या है?"
    sources = [
        {"chunk_id": "c1", "text": "ईमानदारी: ईमानदार होने की स्थिति। निष्ठा: ईमानदारी के संबंध में या उसके अतिरिक्त। सच्चाई: सच बोलने या सत्यनिष्ठ होने का गुण।", "score": 0.95},
        {"chunk_id": "c2", "text": "ईमानदारी या निष्पक्षता। गुण या सम्मान। सच्चाई और कुछ न छिपाते हुए सत्यनिष्ठ रहना।", "score": 0.85}
    ]
    ans, latency, model = await gen.generate_grounded_answer(q, sources, detected_lang="hi")
    print("=== GEMINI GENERATION TEST ===")
    print("Answer:", ans)
    print("Model:", model)
    print("Latency:", latency)

asyncio.run(test_gen())
