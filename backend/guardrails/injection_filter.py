import re
from typing import Tuple, List

# Patterns commonly used in prompt-injection and jailbreaking attacks
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|instructions|secret)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"pwned|format\s+c:|rm\s+-rf", re.IGNORECASE),
    re.compile(r"drop\s+database|<script>|javascript:", re.IGNORECASE),
    re.compile(r"पिछले\s+सभी\s+निर्देशों\s+को\s+अनदेखा\s+करें", re.IGNORECASE),
    re.compile(r"मागील\s+सर्व\s+सूचना\s+दुर्लक्ष\s+करा", re.IGNORECASE)
]


class InjectionFilter:
    """
    Guards against malicious user queries and adversarial passages
    attempting to hijack system instructions.
    """

    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        Scans input for known injection and adversarial patterns.
        Returns: (is_safe, sanitized_text)
        """
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                return False, "[SANITIZED: Potential prompt injection pattern detected]"

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
