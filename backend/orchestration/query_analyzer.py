import re
from typing import Dict, Any, Tuple


# Script & Lexicon Markers
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
LATIN_REGEX = re.compile(r"[a-zA-Z]")

# Distinct Marathi markers in Latin & Devanagari
MARATHI_MARKERS = {
    "म्हणजे", "काय", "आहे", "आहेत", "कसे", "कुठे", "केव्हा", "कोणती", "कोणता", "कोणते",
    "किती", "वेळ", "लागतो", "लागते", "सांगा", "स्पष्ट", "करा", "थोडं", "थोड", "मधील",
    "mhanje", "kay", "ahe", "ahet", "kiti", "vel", "lagto", "lagte", "sanga", "thoda"
}

# Distinct Hindi markers in Latin & Devanagari
HINDI_MARKERS = {
    "क्या", "है", "हैं", "कैसे", "कहाँ", "कब", "कितना", "कितनी", "बताओ", "बताइए",
    "समझाओ", "करना", "करता", "करती", "होता", "होती", "में", "से", "का", "की", "के",
    "kya", "hai", "hain", "kaise", "kahan", "kab", "kitna", "batao", "bataiye", "karta"
}

# Factual query indicators (who, when, what number, how long, longest, largest)
FACTUAL_KEYWORDS = {
    "when", "where", "who", "which", "how many", "how long", "speed", "distance",
    "longest", "largest", "highest", "formula", "equation", "date", "year",
    "कब", "कहाँ", "कौन", "कितना", "कितनी", "लंबाई", "दूरी", "वर्ष", "तारीख",
    "केव्हा", "कुठे", "कोण", "किती", "लांबी", "अंतर", "वर्ष", "दिनांक", "लांब", "मोठी"
}

# Conceptual query indicators (explain, why, how does it work, process, describe)
CONCEPTUAL_KEYWORDS = {
    "why", "how does", "explain", "process", "working", "mechanism", "overview",
    "क्यों", "कैसे काम", "प्रक्रिया", "विस्तार", "समझाइए", "बताइए",
    "का", "कसे काम", "प्रक्रिया", "स्पष्ट करा", "माहिती"
}


class QueryAnalyzer:
    """
    Ultra-low latency (<1ms) heuristic query classifier and language router.
    Avoids expensive LLM round-trips for routing decisions.
    """

    def analyze(self, query: str) -> Dict[str, Any]:
        query_clean = query.strip()
        lang, is_code_mixed, secondary_langs = self.detect_language_and_codemix(query_clean)
        query_type = self.classify_query_type(query_clean)
        strategy = self.select_adaptive_strategy(query_type, is_code_mixed, lang)

        return {
            "query": query_clean,
            "language": lang,
            "is_code_mixed": is_code_mixed,
            "secondary_languages": secondary_langs,
            "query_type": query_type,
            "adaptive_strategy": strategy
        }

    def detect_language_and_codemix(self, text: str) -> Tuple[str, bool, list]:
        has_devanagari = bool(DEVANAGARI_REGEX.search(text))
        has_latin = bool(LATIN_REGEX.search(text))
        tokens = [t.lower() for t in re.findall(r"[\u0900-\u097Fa-zA-Z]+", text)]

        marathi_hits = sum(1 for t in tokens if t in MARATHI_MARKERS)
        hindi_hits = sum(1 for t in tokens if t in HINDI_MARKERS)

        # Code-mixed evaluation
        is_code_mixed = False
        secondary = []

        if has_devanagari and has_latin:
            is_code_mixed = True
            secondary.append("en")

        if marathi_hits > 0 and hindi_hits > 0:
            is_code_mixed = True
            secondary.extend(["hi", "mr"])

        # Determine Primary Language
        if marathi_hits > hindi_hits:
            primary_lang = "mr"
            if has_latin and "en" not in secondary:
                secondary.append("en")
                is_code_mixed = True
        elif hindi_hits > marathi_hits:
            primary_lang = "hi"
            if has_latin and "en" not in secondary:
                secondary.append("en")
                is_code_mixed = True
        elif has_devanagari:
            # Default Devanagari bias (Hindi/Marathi based on marker checks)
            primary_lang = "mr" if "म्हणजे" in text or "आहे" in text or "काय" in text else "hi"
        else:
            # Latin Script
            if marathi_hits > 0:
                primary_lang = "mr"
                is_code_mixed = True
            elif hindi_hits > 0:
                primary_lang = "hi"
                is_code_mixed = True
            else:
                primary_lang = "en"

        return primary_lang, is_code_mixed, list(set(secondary))

    def classify_query_type(self, query: str) -> str:
        q_lower = query.lower()
        
        # Check conceptual markers
        if any(w in q_lower for w in CONCEPTUAL_KEYWORDS):
            return "conceptual"

        # Check factual markers
        if any(w in q_lower for w in FACTUAL_KEYWORDS):
            return "factual"

        # Check exact terminology/entity queries (short queries <= 3 words)
        words = q_lower.split()
        if len(words) <= 3:
            return "exact_entity"

        return "general"

    def select_adaptive_strategy(self, query_type: str, is_code_mixed: bool, language: str) -> Dict[str, Any]:
        """
        Determines the optimal chunking collection target, vector weight, BM25 weight, and retrieval params.
        """
        if query_type == "factual":
            return {
                "target_collection": "passage",
                "vector_weight": 0.60,
                "bm25_weight": 0.40,
                "top_k": 20,
                "rationale": "Factual and definition queries prioritize intact candidate passages and dense semantic vector retrieval."
            }
        elif query_type == "conceptual":
            return {
                "target_collection": "semantic",
                "vector_weight": 0.70,
                "bm25_weight": 0.30,
                "top_k": 20,
                "rationale": "Conceptual query prioritizes semantic topical chunks and dense multilingual vector similarity."
            }
        elif query_type == "exact_entity":
            return {
                "target_collection": "passage",
                "vector_weight": 0.40,
                "bm25_weight": 0.60,
                "top_k": 20,
                "rationale": "Exact entity and terminology queries utilize passage-level BM25 and dense hybrid matching."
            }
        elif is_code_mixed:
            return {
                "target_collection": "combined",
                "vector_weight": 0.65,
                "bm25_weight": 0.35,
                "top_k": 25,
                "rationale": "Code-mixed query utilizes multi-collection dense cross-lingual search with normalized lexical matching."
            }
        else:
            return {
                "target_collection": "sliding",
                "vector_weight": 0.55,
                "bm25_weight": 0.45,
                "top_k": 20,
                "rationale": "Balanced hybrid retrieval with sliding-window contextual chunks."
            }
