"""
cleaner.py - Text cleaning and Unicode normalization pipeline for Indic & English text.
Preserves native Devanagari script (Hindi, Marathi) without transliteration.
"""
import re
import unicodedata
from typing import Optional

class TextCleaner:
    def __init__(self):
        # Regex for excessive whitespace
        self._whitespace_re = re.compile(r'\s+')
        # Regex to strip non-printable control characters while preserving Devanagari Unicode
        self._control_char_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
        # HTML tag stripper
        self._html_tag_re = re.compile(r'<[^>]+>')

    def clean_text(self, text: Optional[str]) -> str:
        """
        Cleans and normalizes text:
        1. Handle null/empty
        2. Unicode normalization (NFC)
        3. Strip HTML tags and control chars
        4. Normalize whitespace
        5. Preserves Devanagari characters (0900-097F) & standard punctuation.
        """
        if text is None:
            return ""
            
        if not isinstance(text, str):
            text = str(text)

        # 1. Unicode NFC Normalization (combines base characters and matras properly)
        text = unicodedata.normalize('NFC', text)

        # 2. Strip HTML artifacts if present
        text = self._html_tag_re.sub(' ', text)

        # 3. Strip control characters
        text = self._control_char_re.sub('', text)

        # 4. Normalize multiple spaces, tabs, newlines to single space
        text = self._whitespace_re.sub(' ', text)

        return text.strip()

    def is_valid_passage(self, text: str, min_length: int = 15) -> bool:
        """
        Check if passage contains meaningful content.
        """
        if not text:
            return False
        if len(text.strip()) < min_length:
            return False
        # Ensure it has alphanumeric or Indic characters
        if not any(c.isalnum() for c in text):
            return False
        return True
