"""
Rule engine: regex-based pattern matching for jailbreak / prompt injection phrases.
"""
import re
from dataclasses import dataclass, field

@dataclass
class RuleResult:
    matched: bool
    patterns: list[str]
    risk_score: float  # 0.0 – 1.0

_RAW_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"pretend\s+(you\s+are|to\s+be).{0,60}?(without\s+restrictions|no\s+restrictions|no\s+limits)",
    r"you\s+are\s+now\s+(DAN|an?\s+AI\s+without\s+restrictions|jailbroken)",
    r"do\s+anything\s+now",
    r"act\s+as\s+(if\s+)?you\s+(have\s+no|don'?t\s+have\s+any)\s+restrictions",
    r"override\s+(your\s+)?(safety|ethical|content)\s+(filters|guidelines|policy)",
    r"disregard\s+(all\s+)?(previous|prior)\s+(instructions|prompts|guidelines)",
    r"enable\s+developer\s+mode",
    r"forget\s+(all\s+)?(your|previous)\s+(instructions|training|guidelines)",
    r"new\s+persona.{0,40}?no\s+restrictions",
    # 5+ additional realistic patterns:
    r"system\s+prompt\s*[:=]",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|prompt)",
    r"(bypass|circumvent|evade)\s+(safety|content|ethical|security)\s+(filter|check|policy|guard)",
    r"respond\s+as\s+(an?\s+)?(unfiltered|unrestricted|uncensored)\s+(AI|model|assistant)",
    r"(sudo|root|admin)\s+mode",
    r"translate\s+the\s+following\s+into\s+base64",
    r"(execute|run)\s+(the\s+)?(following\s+)?(code|command|script)\s*:",
    r"print\s+(your\s+)?(initial|original|full)\s+(prompt|instructions|system)",
    r"token\s+smuggling",
    r"prompt\s+injection",
    r"hypothetically\s+speaking.{0,40}?(how\s+to|ways?\s+to).{0,40}?(harm|kill|attack|hack)",
    r"as\s+a\s+(fictional|hypothetical)\s+character.{0,40}?(without\s+restrictions|no\s+limits)",
    r"output\s+(must|should)\s+(not\s+)?include\s+(any\s+)?(filter|restriction|safety)",
    r"(switch|change)\s+(to\s+)?unrestricted\s+mode",
]

class RuleEngine:
    """Scans text against a battery of compiled jailbreak / injection regex patterns."""

    def __init__(self) -> None:
        self._compiled: list[tuple[str, re.Pattern]] = [
            (raw, re.compile(raw, re.IGNORECASE | re.DOTALL))
            for raw in _RAW_PATTERNS
        ]

    def scan(self, text: str) -> RuleResult:
        """
        Scan text against all patterns.
        Returns RuleResult with matched flag, list of matching pattern strings, and risk_score.
        risk_score is proportional to the number of matches (capped at 1.0).
        """
        matched_patterns: list[str] = []
        for raw, pattern in self._compiled:
            if pattern.search(text):
                matched_patterns.append(raw)
        
        n = len(matched_patterns)
        total = len(self._compiled)
        # Score: 0 matches → 0.0; 1 match → ~0.15; scales up, max 1.0
        risk_score = min(1.0, n / max(total * 0.15, 1))
        
        return RuleResult(
            matched=bool(matched_patterns),
            patterns=matched_patterns,
            risk_score=risk_score,
        )
