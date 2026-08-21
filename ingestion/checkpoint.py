"""
checkpoint.py - Resumable ingestion state manager.
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class IngestionCheckpoint:
    def __init__(self, checkpoint_file: str = "./checkpoints/ingestion_state.json"):
        self.checkpoint_file = checkpoint_file
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_file)), exist_ok=True)
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint file: {e}. Starting fresh.")
        return {
            "last_updated": None,
            "languages": {},
            "total_records_processed": 0,
            "total_chunks_created": 0,
            "total_embeddings_created": 0,
            "qdrant_points_uploaded": 0
        }

    def save(self):
        import datetime
        self.state["last_updated"] = datetime.datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def update_language_progress(
        self,
        lang: str,
        split: str,
        records_processed: int,
        chunks_created: int,
        status: str = "in_progress"
    ):
        if lang not in self.state["languages"]:
            self.state["languages"][lang] = {}
        self.state["languages"][lang][split] = {
            "records_processed": records_processed,
            "chunks_created": chunks_created,
            "status": status
        }
        self.save()

    def get_language_progress(self, lang: str, split: str) -> Dict[str, Any]:
        return self.state.get("languages", {}).get(lang, {}).get(split, {})

    def reset(self):
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        self.state = self._load()
