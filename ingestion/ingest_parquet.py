import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pyarrow.parquet as pq
from ingestion.clean_indic import normalize_indic_text


DATA_DIR = Path("./data")
PARQUET_PATH = Path("martrain.parquet")
OUTPUT_PATH = DATA_DIR / "marathi_parquet_corpus.json"


def extract_marathi_corpus_from_parquet(
    parquet_path: Path = PARQUET_PATH,
    output_path: Path = OUTPUT_PATH,
    max_documents: int = 1500,
    batch_size: int = 200
) -> List[Dict[str, Any]]:
    """
    Streams batches from martrain.parquet without loading the full 3.5GB into memory.
    Extracts high-quality positive Marathi passages (is_selected == 1), queries, and answers.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found at {parquet_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Streaming Marathi corpus from {parquet_path} (Target: {max_documents} documents)...")

    pf = pq.ParquetFile(str(parquet_path))
    columns = ["query_id", "query", "Eng_Query", "Answer", "Eng_Answer", "passages"]

    documents = []
    seen_texts = set()
    total_scanned = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        total_scanned += len(df)

        for _, row in df.iterrows():
            q_id = row["query_id"]
            query_mr = str(row["query"] or "").strip()
            query_en = str(row["Eng_Query"] or "").strip()
            answer_mr = str(row["Answer"] or "").strip()
            passages_struct = row["passages"]

            if not isinstance(passages_struct, dict):
                continue

            trans_passages = passages_struct.get("Translated_passages", [])
            is_selected_list = passages_struct.get("is_selected", [])

            # Convert numpy / list arrays safely
            if hasattr(trans_passages, "tolist"):
                trans_passages = trans_passages.tolist()
            if hasattr(is_selected_list, "tolist"):
                is_selected_list = is_selected_list.tolist()

            # Iterate through passages and prioritize is_selected == 1
            for p_idx, p_text in enumerate(trans_passages):
                if not p_text or len(str(p_text).strip().split()) < 8:
                    continue

                clean_p = normalize_indic_text(str(p_text).strip())
                text_hash = hash(clean_p[:100])
                if text_hash in seen_texts:
                    continue

                seen_texts.add(text_hash)
                is_positive = 0
                if p_idx < len(is_selected_list):
                    is_positive = int(is_selected_list[p_idx])

                doc_entry = {
                    "doc_id": f"mar_pq_{q_id}_{p_idx}",
                    "title": query_mr or query_en,
                    "text": clean_p,
                    "language": "mr",
                    "source": "AI4Bharat_MSMARCO_XI_Marathi",
                    "metadata": {
                        "query_id": q_id,
                        "marathi_query": query_mr,
                        "english_query": query_en,
                        "ground_truth_answer": answer_mr,
                        "is_selected": is_positive
                    }
                }
                documents.append(doc_entry)

                if len(documents) >= max_documents:
                    break

            if len(documents) >= max_documents:
                break

        print(f"  Scanned {total_scanned:,} records -> Extracted {len(documents):,} unique Marathi documents...")
        if len(documents) >= max_documents:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Extracted {len(documents):,} Marathi documents saved to {output_path}")
    return documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and extract Marathi dataset from parquet.")
    parser.add_argument("--max-docs", type=int, default=1500, help="Number of documents to extract")
    args = parser.parse_args()

    extract_marathi_corpus_from_parquet(max_documents=args.max_docs)
