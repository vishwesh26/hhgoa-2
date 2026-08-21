"""
deduplicator.py - Deterministic SHA-256 passage deduplication.
"""
import hashlib
from typing import Set, Tuple, Dict, Any

class PassageDeduplicator:
    def __init__(self):
        self._seen_hashes: Set[str] = set()
        self.records_seen = 0
        self.records_kept = 0
        self.duplicates_removed = 0

    def compute_hash(self, text: str) -> str:
        """
        Compute SHA-256 fingerprint of normalized text.
        """
        norm = " ".join(text.lower().split())
        return hashlib.sha256(norm.encode('utf-8')).hexdigest()

    def check_and_add(self, text: str) -> Tuple[bool, str]:
        """
        Returns (is_unique: bool, text_hash: str).
        If unique, records it in the deduplication registry.
        """
        self.records_seen += 1
        h = self.compute_hash(text)
        if h in self._seen_hashes:
            self.duplicates_removed += 1
            return False, h
            
        self._seen_hashes.add(h)
        self.records_kept += 1
        return True, h

    def get_stats(self) -> Dict[str, int]:
        return {
            "records_seen": self.records_seen,
            "records_kept": self.records_kept,
            "duplicates_removed": self.duplicates_removed,
            "unique_hashes": len(self._seen_hashes)
        }

    def reset(self):
        self._seen_hashes.clear()
        self.records_seen = 0
        self.records_kept = 0
        self.duplicates_removed = 0
