import json
from pathlib import Path
from typing import Dict, List, Any
from backend.chunking.hierarchical import MultiStrategyChunker
from ingestion.clean_indic import normalize_indic_text


DATA_DIR = Path("./data")
RAW_DATA_PATH = DATA_DIR / "msmarco_xi_corpus.json"
CHUNKS_ALL_PATH = DATA_DIR / "chunks_all.json"
CHUNKS_SENTENCE_PATH = DATA_DIR / "chunks_sentence.json"
CHUNKS_SLIDING_PATH = DATA_DIR / "chunks_sliding.json"
CHUNKS_SEMANTIC_PATH = DATA_DIR / "chunks_semantic.json"


def process_and_chunk_corpus() -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads the normalized corpus, applies multi-strategy chunking,
    and produces metadata-rich indexed artifacts for each strategy.
    """
    if not RAW_DATA_PATH.exists():
        from ingestion.download_msmarco import prepare_msmarco_xi_corpus
        prepare_msmarco_xi_corpus(RAW_DATA_PATH)

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # If Marathi parquet corpus exists, merge it into the dataset
    marathi_parquet_path = DATA_DIR / "marathi_parquet_corpus.json"
    if marathi_parquet_path.exists():
        with open(marathi_parquet_path, "r", encoding="utf-8") as f:
            marathi_docs = json.load(f)
            docs.extend(marathi_docs)
            print(f"[INFO] Merged {len(marathi_docs)} Marathi documents from parquet corpus.")

    chunker = MultiStrategyChunker()
    all_sentence_chunks = []
    all_sliding_chunks = []
    all_semantic_chunks = []
    all_chunks_combined = []

    print(f"[INFO] Processing {len(docs)} total documents through 4 chunking strategies...")

    for doc in docs:
        doc_id = doc["doc_id"]
        lang = doc["language"]
        title = doc.get("title", "")
        clean_text = normalize_indic_text(doc["text"])

        result = chunker.process_document(
            doc_id=doc_id,
            text=clean_text,
            language=lang,
            title=title
        )

        all_sentence_chunks.extend(result["sentence"])
        all_sliding_chunks.extend(result["sliding_window"])
        all_semantic_chunks.extend(result["semantic"])

        all_chunks_combined.extend(result["sentence"])
        all_chunks_combined.extend(result["sliding_window"])
        all_chunks_combined.extend(result["semantic"])

    # Save to disk
    with open(CHUNKS_SENTENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_sentence_chunks, f, ensure_ascii=False, indent=2)

    with open(CHUNKS_SLIDING_PATH, "w", encoding="utf-8") as f:
        json.dump(all_sliding_chunks, f, ensure_ascii=False, indent=2)

    with open(CHUNKS_SEMANTIC_PATH, "w", encoding="utf-8") as f:
        json.dump(all_semantic_chunks, f, ensure_ascii=False, indent=2)

    with open(CHUNKS_ALL_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks_combined, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Multi-strategy chunking completed:")
    print(f"  - Sentence chunks:       {len(all_sentence_chunks)}")
    print(f"  - Sliding window chunks: {len(all_sliding_chunks)}")
    print(f"  - Semantic chunks:       {len(all_semantic_chunks)}")
    print(f"  - Combined chunks:       {len(all_chunks_combined)}")

    return {
        "sentence": all_sentence_chunks,
        "sliding": all_sliding_chunks,
        "semantic": all_semantic_chunks,
        "all": all_chunks_combined
    }


if __name__ == "__main__":
    process_and_chunk_corpus()
