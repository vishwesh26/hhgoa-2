import sys
import os
sys.path.insert(0, os.path.abspath("."))
import argparse
import logging
from ingestion.pipeline import MSMARCOIngestionPipeline

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Build Qdrant & BM25 Indexes for MSMARCO-XI Polyglot Voice RAG Engine"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "hi", "mr"],
        help="Target languages to index (default: en hi mr)"
    )
    parser.add_argument(
        "--chunk-strategies",
        nargs="+",
        default=["sentence", "sliding", "semantic"],
        help="Chunking strategies to apply (default: sentence sliding semantic)"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum raw records to process per language for small sample validation (e.g. 100, 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding and vector upload batch size (default: 32)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for resumable state checkpoints"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./reports",
        help="Directory for saving ingestion JSON reports"
    )

    args = parser.parse_args()

    print(f"Initializing MSMARCO-XI Ingestion Pipeline with:")
    print(f"  Languages:          {args.languages}")
    print(f"  Chunk Strategies:   {args.chunk_strategies}")
    print(f"  Max Records:        {args.max_records or 'FULL'}")
    print(f"  Batch Size:         {args.batch_size}")
    print(f"  Checkpoint Dir:     {args.checkpoint_dir}")
    print(f"  Report Dir:         {args.report_dir}\n")

    pipeline = MSMARCOIngestionPipeline(
        languages=args.languages,
        chunk_strategies=args.chunk_strategies,
        max_records=args.max_records,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        report_dir=args.report_dir
    )

    pipeline.run()

if __name__ == "__main__":
    main()
