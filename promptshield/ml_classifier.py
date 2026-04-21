"""
PromptShield — ML-backed classifier for the premium tier.
==========================================================
Sourced from ``promptshield-ml.zip`` (``serve/semantic_classifier_v2.py``).

Provides a three-level tiered classifier:

1. **ML server** — trained DistilBERT served via HTTP (fastest, most accurate).
   Configure the server URL via the ``ML_SERVER_URL`` environment variable or
   the ``server_url`` constructor argument.
2. **Local HuggingFace pipeline** — no server required, loads model locally.
3. **Keyword-density fallback** — no external dependencies.

Usage::

    from promptshield.ml_classifier import MLClassifier

    clf = MLClassifier()
    result = clf.classify("ignore previous instructions")
    # result.is_injection  → True
    # result.confidence    → 0.97
    # result.method        → "ml_server" | "transformer" | "fallback"

Set ``ML_SERVER_URL`` to the public URL produced by the PromptShield ML server
(``serve/server.py`` in ``promptshield-ml.zip``)::

    export ML_SERVER_URL="https://your-tunnel.trycloudflare.com"
"""
from __future__ import annotations

import os
import re

from promptshield.semantic_classifier import ClassifierResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default base model used by the premium tier local pipeline fallback.
PREMIUM_TIER_MODEL = "distilbert-base-uncased"

_ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://localhost:8000")

_INJECTION_KEYWORDS = [
    "ignore", "forget", "bypass", "override", "jailbreak", "disregard",
    "pretend", "act as", "you are now", "do anything", "no restrictions",
    "system prompt", "reveal", "unrestricted", "developer mode", "dan mode",
    "no ethical", "speak freely", "obey", "instruction", "new task",
    "base64", "encoded payload", "indirect injection",
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MLClassifier:
    """
    Premium-tier, tiered classifier backed by the PromptShield ML server.

    Falls back to a local HuggingFace pipeline, then to keyword-density
    scoring, so it is always available regardless of server connectivity.

    Parameters
    ----------
    server_url:
        URL of the PromptShield ML server.  Defaults to the ``ML_SERVER_URL``
        environment variable (or ``http://localhost:8000`` when unset).  Pass
        ``None`` to skip the server entirely.
    model:
        HuggingFace model ID used when falling back to a local pipeline.
    """

    def __init__(
        self,
        server_url: str | None = _ML_SERVER_URL,
        model: str = PREMIUM_TIER_MODEL,
    ) -> None:
        self._server_url = server_url.rstrip("/") if server_url else None
        self._model_id   = model
        self._pipeline   = None   # lazy-loaded on first use
        self._session    = None   # requests.Session — lazy init

    # ── public API ─────────────────────────────────────────────────────────

    def classify(self, text: str) -> ClassifierResult:
        """Classify *text* as injection or benign with a confidence score."""
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

        # 3. Keyword-density fallback
        return self._fallback_classify(text)

    # ── backends ────────────────────────────────────────────────────────────

    def _server_classify(self, text: str) -> ClassifierResult | None:
        try:
            import requests

            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update({"Content-Type": "application/json"})

            r = self._session.post(
                f"{self._server_url}/classify",
                json={"text": text},
                timeout=2.0,
            )
            r.raise_for_status()
            data = r.json()
            return ClassifierResult(
                is_injection=data["injection"],
                confidence=data["confidence"],
                method="ml_server",
            )
        except Exception:
            return None

    def _load_pipeline(self):
        try:
            from transformers import pipeline  # type: ignore
            return pipeline("text-classification", model=self._model_id)
        except Exception:
            return None

    def _transformer_classify(self, text: str) -> ClassifierResult:
        try:
            out = self._pipeline(text[:512])[0]
            label: str = out["label"].upper()
            score: float = out["score"]
            is_injection_label = label in ("INJECTION", "NEGATIVE")
            is_injection = is_injection_label and score >= 0.5
            confidence = score if is_injection_label else 1.0 - score
            return ClassifierResult(
                is_injection=is_injection,
                confidence=round(confidence, 4),
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
