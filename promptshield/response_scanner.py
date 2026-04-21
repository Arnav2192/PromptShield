"""
Response scanner: detects PII and sensitive data in LLM responses.
Supports redaction of findings.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Finding:
    type: str
    value: str
    start: int
    end: int


@dataclass
class ScanResult:
    clean: bool
    findings: list[Finding]


# ---------------------------------------------------------------------------
# Luhn algorithm
# ---------------------------------------------------------------------------

def _luhn_valid(number: str) -> bool:
    """Return True if *number* (digits only) passes the Luhn check."""
    digits = [int(d) for d in number]
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')),
    ("phone", re.compile(
        r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}\b'
    )),
    ("ssn", re.compile(r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b')),
    ("ipv4", re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    )),
    ("api_key", re.compile(r'\bsk-[A-Za-z0-9]{20,}\b')),
    ("bearer_token", re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]{20,}')),
    ("long_secret", re.compile(r'\b[A-Fa-f0-9]{32,}\b')),
    ("password_kv", re.compile(
        r'(?i)\b(password|passwd|secret|api[_\-]?key|auth[_\-]?token)\s*[:=]\s*\S{8,}',
    )),
]

# Credit card pattern (Luhn-validated separately)
_CC_PATTERN = re.compile(r'\b(?:\d[ \-]?){13,16}\b')


class ResponseScanner:
    """
    Scans LLM responses for PII and sensitive data leakage.

    Detected types: email, phone, ssn, credit_card, api_key, bearer_token,
    long_secret, password_kv, ipv4.
    """

    def scan(self, response: str) -> ScanResult:
        """Return a ScanResult listing all findings in *response*."""
        findings: list[Finding] = []

        # Standard patterns
        for kind, pattern in _PATTERNS:
            for m in pattern.finditer(response):
                findings.append(Finding(
                    type=kind,
                    value=m.group(0),
                    start=m.start(),
                    end=m.end(),
                ))

        # Credit card (Luhn-validated)
        for m in _CC_PATTERN.finditer(response):
            digits = re.sub(r'\D', '', m.group(0))
            if 13 <= len(digits) <= 16 and _luhn_valid(digits):
                findings.append(Finding(
                    type="credit_card",
                    value=m.group(0),
                    start=m.start(),
                    end=m.end(),
                ))

        # Sort by position and deduplicate overlapping findings
        findings.sort(key=lambda f: (f.start, f.end))
        findings = _deduplicate(findings)

        return ScanResult(clean=len(findings) == 0, findings=findings)

    def redact(self, response: str) -> str:
        """Replace all findings in *response* with ``[REDACTED:<type>]`` tokens."""
        result = self.scan(response)
        if result.clean:
            return response

        # Apply replacements in reverse order to preserve offsets
        chars = list(response)
        for finding in reversed(result.findings):
            replacement = list(f"[REDACTED:{finding.type}]")
            chars[finding.start : finding.end] = replacement

        return "".join(chars)


def _deduplicate(findings: list[Finding]) -> list[Finding]:
    """Remove findings that are wholly contained within an earlier finding."""
    out: list[Finding] = []
    last_end = -1
    for f in findings:
        if f.start >= last_end:
            out.append(f)
            last_end = f.end
    return out
