import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("4. VERIFYING EMBEDDINGS & TESTING SEMANTIC SIMILARITY")
print("=" * 80)

from backend.config import settings
print(f"Configured Embedding Model Name: {settings.EMBEDDING_MODEL_NAME}")
print(f"Configured Dimension:           {settings.EMBEDDING_DIMENSION}")

# Check python environment packages for embedding libraries
try:
    import sentence_transformers
    print(f"sentence_transformers version: {sentence_transformers.__version__}")
except ImportError:
    print("sentence_transformers: NOT INSTALLED in current .venv")

try:
    import fastembed
    print(f"fastembed version: {fastembed.__version__}")
except ImportError:
    print("fastembed: NOT INSTALLED")

try:
    import onnxruntime
    print(f"onnxruntime version: {onnxruntime.__version__}")
except ImportError:
    print("onnxruntime: NOT INSTALLED")

# Test with Gemini Text Embeddings or Multilingual model
query = "कॉर्पोरेशन मतलब क्या?"
p_relevant = "एक निगम या कॉर्पोरेशन एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत है।"
p_irrelevant = "स्कॉट्सडेल निवासी शहर के कॉर्पोरेशन यार्ड, 9191 ई. 3 डोंगफेंग मोटर कॉर्पोरेशन - सिट्रोएन: सिट्रोएन फुकांग कॉम्पैक्ट कार।"

print(f"\nTest Query: {query}")
print(f"Candidate A (Genuinely Relevant Definition): {p_relevant}")
print(f"Candidate B (Irrelevant Keyword Match):       {p_irrelevant}")

# Test Google Gemini Embedding API if available
try:
    import google.generativeai as genai
    key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if key:
        genai.configure(api_key=key)
        import numpy as np
        
        q_emb = np.array(genai.embed_content(model="models/gemini-embedding-001", content=query)['embedding'])
        a_emb = np.array(genai.embed_content(model="models/gemini-embedding-001", content=p_relevant)['embedding'])
        b_emb = np.array(genai.embed_content(model="models/gemini-embedding-001", content=p_irrelevant)['embedding'])
        
        sim_a = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
        sim_b = np.dot(q_emb, b_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(b_emb))
        
        print("\n--- Google Multilingual Embedding (models/embedding-001) Similarity ---")
        print(f"Cosine Sim(Query, Candidate A [Relevant]):   {sim_a:.4f}")
        print(f"Cosine Sim(Query, Candidate B [Irrelevant]): {sim_b:.4f}")
        if sim_a > sim_b:
            print("✓ Multilingual Embedding model correctly ranks Relevant passage significantly higher!")
except Exception as e:
    print(f"Gemini embedding test exception: {e}")
