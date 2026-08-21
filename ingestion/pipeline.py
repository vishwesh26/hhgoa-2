"""
pipeline.py - Full end-to-end ingestion and indexing pipeline for MSMARCO-XI.
"""
import logging
from typing import List, Optional
from .dataset_loader import MSMARCODatasetLoader
from .cleaner import TextCleaner
from .deduplicator import PassageDeduplicator
from .chunkers import get_chunker, DocumentChunk
from .embedder import MultilingualEmbedder
from .qdrant_indexer import QdrantIndexer
from .bm25_indexer import BM25Indexer
from .checkpoint import IngestionCheckpoint
from .stats import IngestionStats

logger = logging.getLogger(__name__)

class MSMARCOIngestionPipeline:
    def __init__(
        self,
        languages: List[str] = None,
        chunk_strategies: List[str] = None,
        max_records: Optional[int] = None,
        batch_size: int = 32,
        checkpoint_dir: str = "./checkpoints",
        report_dir: str = "./reports"
    ):
        self.languages = languages or ["en", "hi", "mr"]
        self.chunk_strategies = chunk_strategies or ["sentence", "sliding", "semantic"]
        self.max_records = max_records
        self.batch_size = batch_size
        self.report_dir = report_dir

        self.loader = MSMARCODatasetLoader(target_languages=self.languages)
        self.cleaner = TextCleaner()
        self.deduplicator = PassageDeduplicator()
        self.embedder = MultilingualEmbedder(batch_size=self.batch_size)
        self.qdrant_indexer = QdrantIndexer()
        self.bm25_indexer = BM25Indexer()
        self.checkpoint = IngestionCheckpoint(checkpoint_file=f"{checkpoint_dir}/ingestion_state.json")
        self.stats = IngestionStats()
        self.stats.languages = self.languages

    def run(self):
        logger.info(f"Starting MSMARCO-XI Ingestion for languages={self.languages} (max_records={self.max_records})...")
        
        chunks_by_strategy: dict[str, List[DocumentChunk]] = {
            strat: [] for strat in self.chunk_strategies
        }

        # 1. Stream, clean, deduplicate, and chunk
        for lang in self.languages:
            logger.info(f"Processing language: '{lang}'...")
            
            # Determine source stream
            stream_lang = lang if lang in ["hi", "mr"] else "hi"
            records_stream = self.loader.stream_raw_records(
                lang=stream_lang,
                split="val",
                max_records=self.max_records
            )

            records_count = 0
            for raw_record in records_stream:
                self.stats.add_records_processed(1)
                records_count += 1

                # Extract passages for target language
                passages = self.loader.extract_searchable_passages(raw_record, target_lang=lang)
                if not passages:
                    self.stats.add_records_skipped(1)
                    continue

                for p in passages:
                    cleaned_text = self.cleaner.clean_text(p["text"])
                    if not self.cleaner.is_valid_passage(cleaned_text):
                        self.stats.add_records_skipped(1)
                        continue

                    # Deduplicate
                    is_unique, _ = self.deduplicator.check_and_add(cleaned_text)
                    if not is_unique:
                        self.stats.add_duplicates_removed(1)
                        continue

                    self.stats.add_documents_kept(1)

                    # Apply multiple chunking strategies
                    for strat in self.chunk_strategies:
                        chunker = get_chunker(strat)
                        doc_chunks = chunker.chunk(
                            text=cleaned_text,
                            document_id=p["document_id"],
                            language=lang,
                            metadata={
                                "query_id": p.get("query_id"),
                                "is_selected": p.get("is_selected"),
                                "source": p.get("source", "MSMARCO-XI"),
                                "query_context": p.get("query_context")
                            }
                        )
                        chunks_by_strategy[strat].extend(doc_chunks)
                        self.stats.add_chunks(strat, len(doc_chunks))

        # 2. Embed and Index in Qdrant per strategy
        for strat, chunk_list in chunks_by_strategy.items():
            if not chunk_list:
                continue

            logger.info(f"Embedding & indexing {len(chunk_list)} chunks for strategy='{strat}'...")
            
            # Process in batches
            for i in range(0, len(chunk_list), self.batch_size):
                batch_chunks = chunk_list[i : i + self.batch_size]
                texts = [c.text for c in batch_chunks]
                
                vectors = self.embedder.embed_texts(texts)
                self.stats.add_embeddings(len(vectors))

                pts_count = self.qdrant_indexer.upload_chunk_batch(strat, batch_chunks, vectors)
                self.stats.add_qdrant_points(pts_count)

            # 3. Build persistent BM25 index per strategy
            bm25_count = self.bm25_indexer.build_and_save(strat, chunk_list)
            self.stats.add_bm25_docs(bm25_count)

        # 4. Also build combined index across all strategies for fast single-search mode
        all_chunks = []
        for c_list in chunks_by_strategy.values():
            all_chunks.extend(c_list)
        if all_chunks:
            self.bm25_indexer.build_and_save("combined", all_chunks)

        # 5. Save checkpoint and print summary
        self.checkpoint.state["total_records_processed"] = self.stats.records_processed
        self.checkpoint.state["total_chunks_created"] = sum(self.stats.chunks_by_strategy.values())
        self.checkpoint.state["total_embeddings_created"] = self.stats.embeddings_created
        self.checkpoint.state["qdrant_points_uploaded"] = self.stats.qdrant_points
        self.checkpoint.save()

        self.stats.print_summary()
        return self.stats.generate_report(save_dir=self.report_dir)
