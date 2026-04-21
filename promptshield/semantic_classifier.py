"""
Semantic classifier for prompt injection detection.

Free-tier model: ``deepset/prompt-injections`` — a DistilBERT model fine-tuned
specifically on labeled prompt-injection data.  It outputs ``INJECTION`` /
``BENIGN`` labels and is used as the default for all free-tier PromptShield
instances.

If the model cannot be loaded (offline environment, missing weights), a
keyword-density fallback activates transparently.
"""
from __future__ import annotations
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tier model registry
# ---------------------------------------------------------------------------

#: HuggingFace model used for the free tier.
FREE_TIER_MODEL = "deepset/prompt-injections"

@dataclass
class ClassifierResult:
    is_injection: bool
    confidence: float  # 0.0 – 1.0
    method: str  # "transformer" or "fallback"

_INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "jailbreak",
    "DAN",
    "do anything now",
    "no restrictions",
    "unrestricted",
    "uncensored",
    "bypass safety",
    "override guidelines",
    "pretend you are",
    "you are now",
    "developer mode",
    "disregard",
    "forget your training",
    "prompt injection",
    "token smuggling",
    "act as",
    "hypothetically speaking",
]

class SemanticClassifier:
    """
    Classifies text as injection or benign.

    Uses ``deepset/prompt-injections`` (free tier) by default — a DistilBERT
    model fine-tuned on labeled prompt-injection data that outputs ``INJECTION``
    and ``BENIGN`` labels.  Falls back to keyword-density scoring when the
    transformer pipeline cannot be loaded.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to :data:`FREE_TIER_MODEL`.
    threshold : float
        Minimum confidence required to classify a text as an injection (0–1).
    """

    def __init__(
        self,
        model_name: str = FREE_TIER_MODEL,
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._pipeline = None
        self._try_load_pipeline()

    def _try_load_pipeline(self) -> None:
        try:
            from transformers import pipeline  # type: ignore
            self._pipeline = pipeline("text-classification", model=self.model_name)
        except Exception:
            self._pipeline = None

    def classify(self, text: str) -> ClassifierResult:
        """
        Classify text as injection (True) or benign (False) with a confidence score.
        Falls back to keyword-density scoring if the transformer pipeline is unavailable.
        """
        if self._pipeline is not None:
            try:
                return self._classify_transformer(text)
            except Exception:
                pass
        return self._classify_fallback(text)

    def _classify_transformer(self, text: str) -> ClassifierResult:
        truncated = text[:512]
        result = self._pipeline(truncated)[0]
        label: str = result["label"].upper()
        score: float = result["score"]
        # deepset/prompt-injections outputs INJECTION / BENIGN labels.
        # For any model that still uses NEGATIVE/POSITIVE (e.g. SST-2 fallback),
        # treat NEGATIVE as a proxy for injection so the logic stays safe.
        is_injection_label = label in ("INJECTION", "NEGATIVE")
        is_injection = is_injection_label and score >= self.threshold
        # For injection-positive labels the raw score is the confidence.
        # For benign/positive labels the score represents certainty the text
        # is NOT an injection, so we invert it to get injection confidence.
        confidence = score if is_injection_label else 1.0 - score
        return ClassifierResult(
            is_injection=is_injection,
            confidence=confidence,
            method="transformer",
        )

    def _classify_fallback(self, text: str) -> ClassifierResult:
        """
        Keyword-density scorer: counts injection-related keyword hits normalized by
        total word count to produce a confidence score.
        """
        lower = text.lower()
        word_count = max(len(text.split()), 1)
        hits = sum(1 for kw in _INJECTION_KEYWORDS if kw.lower() in lower)
        # Density-based confidence: each keyword hit adds weight
        raw_confidence = min(1.0, hits / max(word_count * 0.05, 1))
        is_injection = raw_confidence >= self.threshold
        return ClassifierResult(
            is_injection=is_injection,
            confidence=raw_confidence,
            method="fallback",
        )
