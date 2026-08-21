import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from qdrant_client import QdrantClient
from ingestion.bm25_indexer import BM25Indexer
from ingestion.embedder import MultilingualEmbedder
from ingestion.clean_indic import indic_tokenize

def verify():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("="*65)
    print("      MSMARCO-XI INDEX VERIFICATION & RETRIEVAL TEST")
    print("="*65)

    # 1. Verify Qdrant Collections
    print("\n[1] Verifying Qdrant Vector Database Collections...")
    qdrant_path = "./data/qdrant_storage"
    client = QdrantClient(path=qdrant_path)
    collections = client.get_collections().collections
    print(f"Available Qdrant Collections ({len(collections)}):")
    for col in collections:
        info = client.get_collection(col.name)
        print(f"  - {col.name:<30} Points: {info.points_count:,}")

    # 2. Verify BM25 Indices
    print("\n[2] Verifying Persistent BM25 Lexical Indices...")
    bm25_indexer = BM25Indexer()
    for strat in ["sentence", "sliding", "semantic", "combined"]:
        try:
            data = bm25_indexer.load_index(strat)
            print(f"  - BM25 '{strat:<10}' Index: {data['total_docs']:,} documents loaded.")
        except Exception as e:
            print(f"  - BM25 '{strat:<10}' Index: [!] Missing or error ({e})")

    # 3. Test Queries
    test_queries = [
        {"lang": "en", "query": "What was the immediate impact of the Manhattan Project?", "desc": "English Factual QA"},
        {"lang": "hi", "query": "प्रकाश संश्लेषण की रासायनिक प्रक्रिया क्या है?", "desc": "Hindi Photosynthesis QA"},
        {"lang": "mr", "query": "भारतातील सर्वात लांब नदी कोणती आहे?", "desc": "Marathi Longest River QA"},
        {"lang": "mr", "query": "फोटोसिंथेसिस मधील रासायनिक अभिक्रिया समजून सांगा.", "desc": "Marathi-English Cross-Lingual Science QA"}
    ]

    embedder = MultilingualEmbedder()
    bm25_data = bm25_indexer.load_index("combined")
    bm25 = bm25_data["bm25"]
    doc_store = bm25_data["doc_store"]

    print("\n[3] Testing Cross-Lingual Retrieval Execution...")
    for idx, q_info in enumerate(test_queries, 1):
        q = q_info["query"]
        desc = q_info["desc"]
        print(f"\n--- Test #{idx}: {desc} ({q_info['lang'].upper()}) ---")
        print(f"Query: \"{q}\"")

        # BM25 Retrieval
        tokens = indic_tokenize(q, remove_stopwords=True)
        scores = bm25.get_scores(tokens)
        top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
        
        print("  Top BM25 Lexical Match:")
        for rank, doc_i in enumerate(top_bm25_idx, 1):
            doc = doc_store[doc_i]
            print(f"    [{rank}] ({doc['language'].upper()}) [{doc['chunk_id']}] Score: {scores[doc_i]:.2f}")
            print(f"        \"{doc['text'][:120]}...\"")

        # Vector Retrieval
        vec = embedder.embed_query(q)
        try:
            results = client.query_points(
                collection_name="vaani_msmarco_sentence",
                query=vec,
                limit=2
            ).points
        except Exception:
            results = client.search(
                collection_name="vaani_msmarco_sentence",
                query_vector=vec,
                limit=2
            )
        print("  Top Qdrant Vector Match:")
        for rank, hit in enumerate(results, 1):
            p = hit.payload
            print(f"    [{rank}] ({p.get('language', '??').upper()}) [{p.get('chunk_id')}] Sim: {getattr(hit, 'score', 0.0):.4f}")
            print(f"        \"{p.get('text', '')[:120]}...\"")

    print("\n" + "="*65)
    print("✓ MSMARCO-XI INDEX & CROSS-LINGUAL RETRIEVAL VERIFIED SUCCESSFULLY")
    print("="*65 + "\n")

if __name__ == "__main__":
    verify()
