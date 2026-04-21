"""
RAG Firewall: scans retrieved document chunks for injected instructions before
they are handed to the LLM, and applies context spotlighting.

Spotlighting uses randomized delimiter tags (Microsoft MSRC Delimiting
technique) rather than fixed tags so attackers cannot embed the exact tag
names in advance to escape the trusted-context boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from promptshield.spotlighting import Spotlighter

if TYPE_CHECKING:
    from promptshield.rule_engine import RuleEngine, RuleResult


@dataclass
class RAGScanResult:
    safe: bool
    flagged_chunks: list[int]
    chunk_scores: list[float]
    open_tag: str = ""
    close_tag: str = ""


class RAGFirewall:
    """
    Splits documents into overlapping word-chunks, scans each chunk for
    injection patterns, and wraps safe content in randomized spotlighting
    delimiters (Microsoft MSRC Delimiting technique).

    Parameters
    ----------
    chunk_size : int
        Number of words per chunk.
    overlap : int
        Number of words shared between consecutive chunks.
    spotlighter : Spotlighter | None
        Custom :class:`~promptshield.spotlighting.Spotlighter` instance.
        If *None*, a default instance is created automatically.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        spotlighter: Spotlighter | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._spotlighter = spotlighter or Spotlighter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> list[str]:
        """Split *text* into overlapping word-based chunks."""
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def scan_chunks(
        self, chunks: list[str], rule_engine: "RuleEngine"
    ) -> RAGScanResult:
        """Scan each chunk with *rule_engine* and aggregate results."""
        flagged: list[int] = []
        scores: list[float] = []

        for idx, chunk in enumerate(chunks):
            result = rule_engine.scan(chunk)
            scores.append(result.risk_score)
            if result.matched:
                flagged.append(idx)

        return RAGScanResult(
            safe=len(flagged) == 0,
            flagged_chunks=flagged,
            chunk_scores=scores,
        )

    def spotlight(self, text: str) -> tuple[str, str, str]:
        """
        Wrap *text* in randomized spotlighting delimiters.

        Returns ``(delimited_text, open_tag, close_tag)``.
        """
        return self._spotlighter.spotlight_rag_document(text)

    def process_document(
        self, text: str, rule_engine: "RuleEngine"
    ) -> tuple[str, RAGScanResult]:
        """
        Full pipeline: chunk → scan → spotlight safe content.

        Returns the spotlighted (safe) text and the scan result.
        Flagged chunks are omitted from the spotlighted output.
        The :attr:`RAGScanResult.open_tag` and :attr:`RAGScanResult.close_tag`
        fields contain the randomized delimiter tags used so callers can embed
        the tag names in the LLM system message.
        """
        chunks = self.chunk_text(text)
        scan_result = self.scan_chunks(chunks, rule_engine)

        safe_chunks = [
            chunk
            for idx, chunk in enumerate(chunks)
            if idx not in scan_result.flagged_chunks
        ]
        safe_text = "\n\n".join(safe_chunks)
        spotlighted, open_tag, close_tag = self.spotlight(safe_text)

        scan_result.open_tag = open_tag
        scan_result.close_tag = close_tag

        return spotlighted, scan_result
