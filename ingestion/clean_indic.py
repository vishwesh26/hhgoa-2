import re
import unicodedata
from typing import List, Set

# Indic punctuation marks and danda
DEVANAGARI_DANDA = "\u0964"  # ।
DEVANAGARI_DOUBLE_DANDA = "\u0965"  # ॥

# Common Hindi Stopwords
HINDI_STOPWORDS: Set[str] = {
    "है", "हैं", "का", "के", "की", "को", "में", "से", "पर", "और", "या", "यह", "वह",
    "तो", "भी", "ने", "लिए", "था", "थे", "थी", "होता", "होती", "होते", "किया", "किए",
    "गया", "गए", "गई", "इस", "उस", "एक", "दो", "इन", "उन", "कर", "रहा", "रहे", "रही",
    "क्या", "क्यों", "कैसे", "कहाँ", "कब", "किस", "किसे", "हुए", "हुआ", "हुई",
    "मुझे", "बताओ", "बताइए", "समझाओ", "समझा", "दीजिए", "दो"
}

# Common Marathi Stopwords
MARATHI_STOPWORDS: Set[str] = {
    "आहे", "आहेत", "चा", "ची", "चे", "च्या", "ला", "ना", "मध्ये", "वरून", "आणि", "किंवा",
    "हे", "तो", "ती", "ते", "सुद्धा", "ने", "साठी", "होता", "होते", "होती", "केला", "केले",
    "गेला", "गेले", "या", "त्या", "एक", "दोन", "यांचे", "त्यांचे", "करून", "आहेस",
    "काय", "का", "कसे", "कुठे", "केव्हा", "कोण", "कोणाला", "झाले", "झाला", "झाली",
    "मधील", "मला", "सांग", "सांगा", "समजून", "सांगणे", "करा", "करणे", "माहिती", "द्या"
}

ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "of", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now", "is", "are", "was", "were",
    "me", "tell", "explain", "give"
}

INDIC_SYNONYMS = {
    "फोटोसिंथेसिस": ["photosynthesis", "प्रकाश", "संश्लेषण"],
    "photosynthesis": ["photosynthesis", "प्रकाश", "संश्लेषण", "chlorophyll"],
    "रिएक्शन": ["reaction", "रासायनिक", "अभिक्रिया", "समीकरण"],
    "reaction": ["reaction", "रासायनिक", "समीकरण", "अभिक्रिया"],
    "chemical": ["chemical", "रासायनिक"],
    "केमिकल": ["chemical", "रासायनिक"],
    "मॅनहॅटन": ["manhattan"],
    "मैनहट्टन": ["manhattan"],
    "manhattan": ["manhattan"],
    "ऑक्सिजन": ["oxygen", "प्राणवायू"],
    "oxygen": ["oxygen", "ऑक्सिजन", "प्राणवायू"],
    "प्रोजेक्ट": ["project", "प्रकल्प"],
    "project": ["project", "प्रकल्प"]
}


def normalize_indic_text(text: str) -> str:
    """
    Applies Unicode NFKC normalization, removes control characters,
    normalizes multiple spaces and standardizes Indic dandas.
    """
    if not text:
        return ""

    # Normalize Unicode (e.g. Nukta, combining characters)
    text = unicodedata.normalize("NFKC", text)

    # Standardize whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize double dandas or repeated punctuation
    text = re.sub(r"[।]{2,}", DEVANAGARI_DOUBLE_DANDA, text)
    text = re.sub(r"[.]{2,}", "...", text)

    return text


def indic_tokenize(text: str, remove_stopwords: bool = False, lang: str = "auto", expand_synonyms: bool = True) -> List[str]:
    """
    Tokenizes text containing English, Hindi (Devanagari), Marathi (Devanagari),
    or code-mixed Romanized Indic terms into clean tokens for BM25 indexing and querying.
    """
    text = normalize_indic_text(text).lower()

    # Extract words preserving Latin and Devanagari script blocks (\u0900-\u097F)
    tokens = re.findall(r"[\u0900-\u097Fa-zA-Z0-9_]+", text)

    if not remove_stopwords:
        expanded = []
        for t in tokens:
            expanded.append(t)
            if expand_synonyms and t in INDIC_SYNONYMS:
                expanded.extend(INDIC_SYNONYMS[t])
        return expanded

    stopwords = set()
    if lang == "hi" or lang == "auto":
        stopwords.update(HINDI_STOPWORDS)
    if lang == "mr" or lang == "auto":
        stopwords.update(MARATHI_STOPWORDS)
    if lang == "en" or lang == "auto":
        stopwords.update(ENGLISH_STOPWORDS)

    filtered = [t for t in tokens if t not in stopwords and len(t) > 1]
    
    if not expand_synonyms:
        return filtered

    expanded = []
    for t in filtered:
        expanded.append(t)
        if t in INDIC_SYNONYMS:
            expanded.extend([s for s in INDIC_SYNONYMS[t] if s not in stopwords])

    return list(dict.fromkeys(expanded))
