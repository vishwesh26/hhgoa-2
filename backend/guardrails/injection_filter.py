import re
from typing import Tuple, List

# Comprehensive Multilingual Patterns for Prompt-Injection, Jailbreaking, and Adversarial Probes
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|other)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*(override|reveal|leak|show|display)", re.IGNORECASE),
    re.compile(r"(reveal|print|show|repeat|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions|secret|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|unrestricted|jailbroken|an\s+ai\s+without\s+rules|developer\s+mode)", re.IGNORECASE),
    re.compile(r"(pretend|act\s+as|simulate\s+being)\s+(an\s+unfiltered|an\s+evil|a\s+hacked)", re.IGNORECASE),
    re.compile(r"pwned|format\s+c:|rm\s+-rf|sudo\s+rm|drop\s+database|<script>|javascript:", re.IGNORECASE),
    re.compile(r"bypass\s+(guardrails|safety|filters|rules)", re.IGNORECASE),
    # Hindi patterns
    re.compile(r"पिछले\s+सभी\s+(निर्देशों|आदेशों)\s+को\s+(अनदेखा|भूल)\s*(करें|जाओ)", re.IGNORECASE),
    re.compile(r"(सिस्टम\s+प्रॉम्प्ट|गुप्त\s+नियम|निर्देश)\s*(दिखाओ|बताओ|प्रकट\s+करो)", re.IGNORECASE),
    re.compile(r"सुरक्षा\s+नियम\s*(हटाओ|तोड़ो|बायपास)", re.IGNORECASE),
    # Marathi patterns
    re.compile(r"मागील\s+सर्व\s+सूचना\s*(दुर्लक्ष\s+करा|विसरा)", re.IGNORECASE),
    re.compile(r"(सिस्टम\s+प्रॉम्प्ट|सूचना|नियम)\s*(दाखवा|सांगा|उघड\s+करा)", re.IGNORECASE),
    re.compile(r"सुरक्षा\s*(बायपास\s+करा|काढून\s+टाका)", re.IGNORECASE)
]


class InjectionFilter:
    """
    Guards against malicious user queries, jailbreaks, and adversarial passages
    attempting to hijack system instructions or exfiltrate prompts.
    """

    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        Scans input for known injection and adversarial patterns.
        Returns: (is_safe, sanitized_text)
        """
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                return False, "[SANITIZED: Potential prompt injection or jailbreak detected]"

        return True, text

    def sanitize_retrieved_passage(self, passage: str) -> str:
        """
        Neutralizes directive keywords inside retrieved passages so they cannot
        act as executable commands for the LLM.
        """
        sanitized = passage
        for pattern in INJECTION_PATTERNS:
            sanitized = pattern.sub("[UNTRUSTED_INSTRUCTION_REMOVED]", sanitized)
        return sanitized
