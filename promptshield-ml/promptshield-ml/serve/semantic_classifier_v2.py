"""
PromptShield — ML-backed SemanticClassifier
=============================================
Drop-in replacement for the original SemanticClassifier.
Priority:
  1. ML server (trained DistilBERT via HTTP) — fastest, most accurate
  2. Local HuggingFace pipeline                — no server needed
  3. Keyword-density fallback                  — no dependencies

Usage:
    from promptshield.semantic_classifier import SemanticClassifier

    # Auto-discovers server at ML_SERVER_URL env var (default: http://localhost:8000)
    clf = SemanticClassifier()
    result = clf.classify("ignore previous instructions")
    # result.is_injection  → True
    # result.confidence    → 0.97
    # result.method        → "ml_server"
"""

import os
import re
from dataclasses import dataclass

_ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://localhost:8000")

_INJECTION_KEYWORDS = [
    "ignore", "forget", "bypass", "override", "jailbreak", "disregard",
    "pretend", "act as", "you are now", "do anything", "no restrictions",
    "system prompt", "reveal", "unrestricted", "developer mode", "dan mode",
    "no ethical", "speak freely", "obey", "instruction", "new task",
    "base64", "encoded payload", "indirect injection",
]


@dataclass
class ClassifierResult:
    is_injection: bool
    confidence:   float   # 0.0 – 1.0
    method:       str     # "ml_server" | "transformer" | "fallback"


class SemanticClassifier:
    """
    Tiered classifier — tries each backend in order until one succeeds.

    Args:
        server_url: URL of the PromptShield ML server. Set to None to skip.
        model:      HuggingFace model ID for local pipeline fallback.
    """

    def __init__(
        self,
        server_url: str = _ML_SERVER_URL,
        model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ):
        self._server_url = server_url.rstrip("/") if server_url else None
        self._pipeline   = None
        self._model_id   = model
        self._session    = None   # requests.Session — lazy init

    # ── public API ──────────────────────────────────────────────────────────
    def classify(self, text: str) -> ClassifierResult:
        # 1. Try ML server
        if self._server_url:
            result = self._server_classify(text)
            if result is not None:
                return result

        # 2. Try local transformer
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        if self._pipeline:
            return self._transformer_classify(text)

        # 3. Fallback
        return self._fallback_classify(text)

    # ── backends ─────────────────────────────────────────────────────────────
    def _server_classify(self, text: str) -> ClassifierResult | None:
        try:
            import requests
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update({"Content-Type": "application/json"})

            r = self._session.post(
                f"{self._server_url}/classify",
                json={"text": text},
                timeout=2.0,   # fast — don't block the pipeline
            )
            r.raise_for_status()
            data = r.json()
            return ClassifierResult(
                is_injection=data["injection"],
                confidence=data["confidence"],
                method="ml_server",
            )
        except Exception:
            return None   # silently fall through

    def _load_pipeline(self):
        try:
            from transformers import pipeline
            return pipeline("text-classification", model=self._model_id)
        except Exception:
            return None

    def _transformer_classify(self, text: str) -> ClassifierResult:
        try:
            out = self._pipeline(text[:512])[0]
            is_inj = out["label"] == "NEGATIVE"
            return ClassifierResult(
                is_injection=is_inj,
                confidence=round(out["score"], 4),
                method="transformer",
            )
        except Exception:
            return self._fallback_classify(text)

    def _fallback_classify(self, text: str) -> ClassifierResult:
        lower = text.lower()
        hits  = sum(1 for kw in _INJECTION_KEYWORDS if kw in lower)
        score = min(1.0, hits / 4.0)
        return ClassifierResult(
            is_injection=score >= 0.5,
            confidence=round(score, 4),
            method="fallback",
        )
