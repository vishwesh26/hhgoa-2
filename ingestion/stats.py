"""
stats.py - Ingestion metrics calculation and report generation.
"""
import os
import json
import time
import datetime
from typing import Dict, Any, List

class IngestionStats:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.languages: List[str] = []
        self.records_processed = 0
        self.records_skipped = 0
        self.duplicates_removed = 0
        self.documents_kept = 0
        self.chunks_by_strategy: Dict[str, int] = {}
        self.embeddings_created = 0
        self.qdrant_points = 0
        self.bm25_documents = 0

    def add_records_processed(self, count: int = 1):
        self.records_processed += count

    def add_records_skipped(self, count: int = 1):
        self.records_skipped += count

    def add_duplicates_removed(self, count: int = 1):
        self.duplicates_removed += count

    def add_documents_kept(self, count: int = 1):
        self.documents_kept += count

    def add_chunks(self, strategy: str, count: int):
        self.chunks_by_strategy[strategy] = self.chunks_by_strategy.get(strategy, 0) + count

    def add_embeddings(self, count: int):
        self.embeddings_created += count

    def add_qdrant_points(self, count: int):
        self.qdrant_points += count

    def add_bm25_docs(self, count: int):
        self.bm25_documents += count

    def get_elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def estimate_storage_bytes(self) -> Dict[str, Any]:
        """
        Estimates memory and disk storage requirements.
        - Embedding vector: 768 dimensions * 4 bytes = 3,072 bytes per vector
        - Text payload: avg 400 bytes per chunk
        - BM25 inverted index: avg 150 bytes per document
        """
        total_chunks = sum(self.chunks_by_strategy.values())
        vector_storage_bytes = self.embeddings_created * (768 * 4)
        payload_storage_bytes = total_chunks * 400
        bm25_storage_bytes = self.bm25_documents * 150
        total_bytes = vector_storage_bytes + payload_storage_bytes + bm25_storage_bytes
        
        return {
            "vectors_mb": round(vector_storage_bytes / (1024 * 1024), 3),
            "payloads_mb": round(payload_storage_bytes / (1024 * 1024), 3),
            "bm25_mb": round(bm25_storage_bytes / (1024 * 1024), 3),
            "total_estimated_mb": round(total_bytes / (1024 * 1024), 3)
        }

    def generate_report(self, save_dir: str = "./reports") -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)
        elapsed_sec = round(self.get_elapsed_seconds(), 2)
        storage = self.estimate_storage_bytes()
        
        report = {
            "title": "MSMARCO-XI INGESTION REPORT",
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": elapsed_sec,
            "languages": self.languages,
            "records_processed": self.records_processed,
            "records_skipped": self.records_skipped,
            "duplicates_removed": self.duplicates_removed,
            "documents_kept": self.documents_kept,
            "chunks_created": {
                "total": sum(self.chunks_by_strategy.values()),
                "by_strategy": self.chunks_by_strategy
            },
            "embeddings_created": self.embeddings_created,
            "qdrant_points_indexed": self.qdrant_points,
            "bm25_documents_indexed": self.bm25_documents,
            "estimated_storage": storage
        }

        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        report_file = os.path.join(save_dir, f"ingestion-{date_str}.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def print_summary(self):
        report = self.generate_report()
        storage = report["estimated_storage"]
        
        print("\n" + "="*65)
        print("          MSMARCO-XI INGESTION REPORT")
        print("="*65)
        print(f" Languages:               {', '.join(self.languages).upper()}")
        print(f" Records processed:       {self.records_processed:,}")
        print(f" Records skipped:         {self.records_skipped:,}")
        print(f" Duplicates removed:      {self.duplicates_removed:,}")
        print(f" Clean Documents:         {self.documents_kept:,}")
        print(f" Total Chunks:            {report['chunks_created']['total']:,}")
        for strat, c_cnt in self.chunks_by_strategy.items():
            print(f"   - {strat.capitalize():<12} chunks:  {c_cnt:,}")
        print(f" Embeddings generated:    {self.embeddings_created:,}")
        print(f" Qdrant points indexed:   {self.qdrant_points:,}")
        print(f" BM25 documents indexed:  {self.bm25_documents:,}")
        print(f" Estimated Storage:       {storage['total_estimated_mb']} MB (Vectors: {storage['vectors_mb']}MB, BM25: {storage['bm25_mb']}MB)")
        print(f" Total Processing Time:   {report['elapsed_seconds']}s")
        print("="*65 + "\n")
