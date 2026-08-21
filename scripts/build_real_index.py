import sys
import os
import time
import json
import logging
import pyarrow.parquet as pq
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("IndexBuilder")

from backend.config import settings
from backend.retrieval.vector_search import get_shared_qdrant_client, get_embedding_model, check_embedding_health
from ingestion.chunkers import get_chunker, DocumentChunk
from ingestion.cleaner import TextCleaner
from ingestion.deduplicator import PassageDeduplicator
from ingestion.bm25_indexer import BM25Indexer
from qdrant_client.http import models

def build_index(max_train_records: int = 150, max_val_records: int = 60):
    logger.info("=" * 80)
    logger.info("BUILDING HIGH-PERFORMANCE REAL MULTILINGUAL MSMARCO-XI INDEX")
    logger.info("=" * 80)

    # 1. Health check - FAIL FAST
    health = check_embedding_health()
    logger.info(f"Embedding Health Check: {health}")
    if health["status"] != "READY":
        raise RuntimeError(f"FATAL: Embedding model is not available: {health['error']}")

    embed_model = get_embedding_model()
    qdrant = get_shared_qdrant_client()
    cleaner = TextCleaner()
    bm25_indexer = BM25Indexer()

    # 2. Reset Qdrant Collections for v2 prefix
    strategies = ["passage", "sliding", "semantic", "sentence"]
    for strat in strategies:
        col_name = f"{settings.QDRANT_COLLECTION_PREFIX}_{strat}"
        try:
            if qdrant.collection_exists(col_name):
                logger.info(f"Deleting old/stale collection: {col_name}")
                qdrant.delete_collection(col_name)
        except Exception as e:
            logger.warning(f"Note on deleting collection {col_name}: {e}")

        logger.info(f"Creating fresh collection: {col_name} (dim={settings.EMBEDDING_DIMENSION})")
        qdrant.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE
            )
        )

    # 3. Stream Knowledge Records from hintrain.parquet and hinval.parquet
    records_to_process = []
    
    # Load from hintrain.parquet
    if os.path.exists("hintrain.parquet"):
        logger.info(f"Loading {max_train_records} training knowledge records from hintrain.parquet...")
        pf_train = pq.ParquetFile("hintrain.parquet")
        count = 0
        for batch in pf_train.iter_batches(batch_size=50, columns=['query_id', 'query', 'Eng_Query', 'Answer', 'passages', 'query_type']):
            for r in batch.to_pylist():
                records_to_process.append(r)
                count += 1
                if count >= max_train_records:
                    break
            if count >= max_train_records:
                break
        logger.info(f"Loaded {count} records from hintrain.parquet.")

    # Load from hinval.parquet (contains query_id=1102432 at row 0)
    if os.path.exists("hinval.parquet"):
        logger.info(f"Loading {max_val_records} validation knowledge records from hinval.parquet...")
        pf_val = pq.ParquetFile("hinval.parquet")
        count = 0
        for batch in pf_val.iter_batches(batch_size=50, columns=['query_id', 'query', 'Eng_Query', 'Answer', 'passages', 'query_type']):
            for r in batch.to_pylist():
                records_to_process.append(r)
                count += 1
                if count >= max_val_records:
                    break
            if count >= max_val_records:
                break
        logger.info(f"Loaded {count} records from hinval.parquet.")

    # 4. Extract passages, clean, and chunk
    chunks_by_strategy: Dict[str, List[DocumentChunk]] = {strat: [] for strat in strategies}
    doc_count = 0
    lang_dist = {"hi": 0, "en": 0, "mr": 0}

    for r_idx, r in enumerate(records_to_process):
        q_id = r.get("query_id")
        passages = r.get("passages", {})
        if not isinstance(passages, dict):
            continue

        hi_passages = passages.get("Translated_passages", [])
        en_passages = passages.get("English_passages", [])
        is_sel = passages.get("is_selected", [0] * len(hi_passages))

        # Index Hindi candidate passages
        for p_idx, (p_text, sel) in enumerate(zip(hi_passages, is_sel)):
            if not p_text:
                continue
            cleaned = cleaner.clean_text(str(p_text))
            if not cleaner.is_valid_passage(cleaned):
                continue

            doc_id = f"msmarco_hi_{q_id}_p{p_idx}"
            doc_count += 1
            lang_dist["hi"] += 1

            meta = {
                "query_id": q_id,
                "passage_index": p_idx,
                "language": "hi",
                "source": "MSMARCO-XI",
                "is_selected": sel
            }

            for strat in strategies:
                chunker = get_chunker(strat)
                c_list = chunker.chunk(cleaned, doc_id, "hi", meta)
                chunks_by_strategy[strat].extend(c_list)

        # Index selected English candidate passages for cross-lingual support
        for p_idx, (p_text, sel) in enumerate(zip(en_passages, is_sel)):
            if not p_text or sel != 1:
                continue
            cleaned = cleaner.clean_text(str(p_text))
            if not cleaner.is_valid_passage(cleaned):
                continue
            doc_id = f"msmarco_en_{q_id}_p{p_idx}"
            doc_count += 1
            lang_dist["en"] += 1
            meta = {
                "query_id": q_id,
                "passage_index": p_idx,
                "language": "en",
                "source": "MSMARCO-XI",
                "is_selected": sel
            }
            for strat in strategies:
                chunker = get_chunker(strat)
                c_list = chunker.chunk(cleaned, doc_id, "en", meta)
                chunks_by_strategy[strat].extend(c_list)

    logger.info(f"Extracted {doc_count} passages across strategies.")
    for strat, c_list in chunks_by_strategy.items():
        logger.info(f"  - Strategy '{strat}': {len(c_list)} chunks")

    # 5. Embed and Upload to Qdrant & Build BM25 in Streaming Batches
    batch_size = 64
    for strat, chunk_list in chunks_by_strategy.items():
        if not chunk_list:
            continue
        col_name = f"{settings.QDRANT_COLLECTION_PREFIX}_{strat}"
        logger.info(f"Starting batched embedding and upload for strategy='{strat}' ({len(chunk_list)} chunks)...")
        
        t0 = time.perf_counter()
        total_chunks = len(chunk_list)

        for b_idx in range(0, total_chunks, batch_size):
            batch = chunk_list[b_idx : b_idx + batch_size]
            texts = [c.text for c in batch]
            
            raw_vecs = list(embed_model.embed(texts, batch_size=batch_size))
            vectors = []
            for v in raw_vecs:
                norm = np.linalg.norm(v)
                vectors.append((v / norm).tolist() if norm > 0 else v.tolist())

            points = []
            for sub_idx, (chunk, vec) in enumerate(zip(batch, vectors)):
                pt_id = b_idx + sub_idx
                points.append(
                    models.PointStruct(
                        id=pt_id,
                        vector=vec,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "text": chunk.text,
                            "language": chunk.language,
                            "chunk_strategy": chunk.chunk_strategy,
                            "source": chunk.source,
                            **chunk.metadata
                        }
                    )
                )

            qdrant.upsert(
                collection_name=col_name,
                points=points
            )

            processed = min(b_idx + len(batch), total_chunks)
            if processed % 256 == 0 or processed == total_chunks:
                logger.info(f"  [{strat}] Indexed {processed}/{total_chunks} chunks...")

        strat_time = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Completed Qdrant index for '{strat}' in {strat_time/1000:.2f}s.")

        # Build BM25 index for strategy
        logger.info(f"Building BM25 index for strategy='{strat}'...")
        bm25_indexer.build_and_save(strat, chunk_list)

    # 6. Combined BM25 index
    all_chunks = []
    for c_list in chunks_by_strategy.values():
        all_chunks.extend(c_list)
    logger.info(f"Building Combined BM25 index ({len(all_chunks)} chunks)...")
    bm25_indexer.build_and_save("combined", all_chunks)

    # 7. Write Index Metadata
    meta_info = {
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "document_count": doc_count,
        "chunk_counts": {strat: len(c_list) for strat, c_list in chunks_by_strategy.items()},
        "language_distribution": lang_dist,
        "collection_prefix": settings.QDRANT_COLLECTION_PREFIX,
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    os.makedirs("./data", exist_ok=True)
    with open("./data/index_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)

    logger.info("=" * 80)
    logger.info(f"SUCCESS: Real Multilingual Index Built! Metadata:\n{json.dumps(meta_info, indent=2, ensure_ascii=False)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    build_index()
