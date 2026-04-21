"""
PromptShield — main firewall facade.

Orchestrates deobfuscation → rule scanning → semantic classification →
session state tracking for inputs, and RAG document scanning and response
PII scanning for outputs.

Also provides :meth:`PromptShield.build_prompt` — a production-ready
helper that assembles an LLM request with:
  * The Microsoft Safety System Message as the highest-priority system
    instruction (cannot be overridden by user content).
  * The user input isolated via Microsoft MSRC Delimiting spotlighting
    (randomized tags) so indirect prompt injections are neutralized.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

from promptshield.deobfuscator import Deobfuscator
from promptshield.rule_engine import RuleEngine, RuleResult
from promptshield.semantic_classifier import SemanticClassifier, ClassifierResult, FREE_TIER_MODEL
from promptshield.ml_classifier import MLClassifier, PREMIUM_TIER_MODEL
from promptshield.state_manager import StateManager
from promptshield.rag_firewall import RAGFirewall, RAGScanResult
from promptshield.response_scanner import ResponseScanner, ScanResult
from promptshield.safety_system_message import SafetySystemMessage
from promptshield.spotlighting import Spotlighter

# ---------------------------------------------------------------------------
# Tier → model mapping
# ---------------------------------------------------------------------------

#: Maps tier names to their default classifier model.
#: ``"premium"`` uses the ML server-backed :class:`~promptshield.ml_classifier.MLClassifier`.
TIER_MODELS: dict[str, str] = {
    "free":    FREE_TIER_MODEL,
    "premium": PREMIUM_TIER_MODEL,
}


@dataclass
class InputResult:
    blocked: bool
    risk_score: float
    cumulative_score: float
    decoded_text: str
    rule_result: RuleResult
    classifier_result: ClassifierResult
    reason: str


@dataclass
class PromptBundle:
    """
    A fully assembled, security-hardened LLM prompt bundle.

    Attributes
    ----------
    system_message : str
        The safety system message (highest-priority instruction) that must
        be passed as the ``system`` role in the LLM API call.
    user_message : str
        The user input wrapped in randomized spotlighting delimiters.
    open_tag : str
        The randomized delimiter tag that opens the untrusted user section.
    close_tag : str
        The randomized delimiter tag that closes the untrusted user section.
    """

    system_message: str
    user_message: str
    open_tag: str
    close_tag: str


class PromptShield:
    """
    Hybrid, stateful LLM security firewall.

    Follows Microsoft enterprise design patterns:

    * **Safety System Message** — every call to :meth:`build_prompt`
      prepends the Microsoft Safety System Message so harmful-content
      rules are always the highest-priority instruction.
    * **Spotlighting (Delimiting)** — user inputs are wrapped in
      cryptographically random delimiter tags so indirect prompt
      injections embedded in user text cannot escape the untrusted zone.
    * **Inspection pipeline** — deobfuscation → rule engine → semantic
      classifier → stateful session scoring → block/allow decision.
    * **RAG firewall** — retrieved documents are chunk-scanned and
      spotlighted before being injected into the LLM context.
    * **Response scanner** — PII is detected and optionally redacted
      from LLM outputs.

    Parameters
    ----------
    session_id : str
        Identifier for the conversation session (used for stateful scoring).
    tier : str
        Service tier.  ``"free"`` (default) uses the
        ``deepset/prompt-injections`` DistilBERT model.  ``"premium"`` uses
        the ML server-backed :class:`~promptshield.ml_classifier.MLClassifier`
        (DistilBERT fine-tuned on real injection data, served via ONNX + FastAPI).
    block_threshold : float
        Cumulative score at which a request is blocked (default 1.5).
    redact_responses : bool
        If True, ``inspect_response`` automatically redacts PII (default True).
    decay : float
        Decay factor for the state manager.
    classifier_model : str
        HuggingFace model name for the semantic classifier.  Defaults to the
        model associated with the chosen *tier*.  For the premium tier this
        sets the local fallback model (used when the ML server is unavailable).
    ml_server_url : str | None
        URL of the PromptShield ML server used by the premium tier.  Defaults
        to the ``ML_SERVER_URL`` environment variable.  Pass ``None`` to
        disable server calls and use the local pipeline or fallback directly.
    task_instructions : str
        Optional task-specific instructions appended to the safety system
        message.  Safety rules always take precedence over these.
    immediate_block_rule_severity : float
        Rule-engine risk score at or above which the request is blocked
        immediately (before accumulating session score).  Default 0.6,
        which corresponds to roughly three or more matched jailbreak patterns.
        Set to 1.1 to disable immediate rule-based blocking.
    immediate_block_classifier_confidence : float
        Classifier confidence at or above which an injection-labelled result
        causes an immediate block.  Default 0.85.  Applies to both free and
        premium tier classifiers.  Set to 1.1 to disable.
    """

    def __init__(
        self,
        session_id: str,
        tier: str = "free",
        block_threshold: float = 1.5,
        redact_responses: bool = True,
        decay: float = 0.6,
        classifier_model: str | None = None,
        ml_server_url: str | None = None,
        task_instructions: str = "",
        immediate_block_rule_severity: float = 0.6,
        immediate_block_classifier_confidence: float = 0.85,
    ) -> None:
        self.session_id = session_id
        self.tier = tier
        self.redact_responses = redact_responses
        self._immediate_block_rule_severity = immediate_block_rule_severity
        self._immediate_block_classifier_confidence = immediate_block_classifier_confidence

        # Build classifier: premium tier uses the ML-server-backed MLClassifier;
        # all other tiers use the HuggingFace-pipeline SemanticClassifier.
        if tier == "premium":
            server_url = ml_server_url if ml_server_url is not None else os.getenv("ML_SERVER_URL", "http://localhost:8000")
            premium_model = classifier_model if classifier_model is not None else PREMIUM_TIER_MODEL
            self._classifier = MLClassifier(server_url=server_url, model=premium_model)
        else:
            resolved_model = classifier_model if classifier_model is not None else TIER_MODELS.get(tier, FREE_TIER_MODEL)
            self._classifier = SemanticClassifier(model_name=resolved_model)

        self._deobfuscator = Deobfuscator()
        self._rule_engine = RuleEngine()
        self._state = StateManager(decay=decay, threshold=block_threshold)
        self._rag = RAGFirewall()
        self._response_scanner = ResponseScanner()
        self._safety_msg = SafetySystemMessage(
            additional_instructions=task_instructions
        )
        self._spotlighter = Spotlighter()

    # ------------------------------------------------------------------
    # Prompt assembly (Microsoft Safety System Message + Spotlighting)
    # ------------------------------------------------------------------

    def build_prompt(self, user_text: str) -> PromptBundle:
        """
        Assemble a security-hardened LLM prompt bundle.

        The bundle contains:

        1. A **system message** composed of the Microsoft Safety System
           Message (highest-priority, cannot be overridden) followed by
           a spotlight instruction that names the randomized delimiter tags.
        2. A **user message** with the raw user text wrapped in
           per-call randomized delimiters (Microsoft MSRC Delimiting).

        Parameters
        ----------
        user_text : str
            The raw user input to be passed to the LLM.

        Returns
        -------
        PromptBundle
            Use ``bundle.system_message`` as the ``system`` role and
            ``bundle.user_message`` as the ``user`` role in your LLM API
            call.
        """
        # 1. Spotlight the user input — fresh random tags for every call
        spotlighted_user, open_tag, close_tag = self._spotlighter.spotlight_user_input(
            user_text
        )

        # 2. Build spotlight instruction to embed in the system message
        spotlight_instruction = self._spotlighter.build_spotlight_instruction(
            open_tag, close_tag, content_type="user input"
        )

        # 3. Assemble system message: safety policy (highest priority) first,
        #    then the spotlight instruction so the LLM knows the tag names.
        #    Task-specific instructions (if any) are already appended inside
        #    SafetySystemMessage.build() at the lowest priority.
        safety_text = self._safety_msg.build()
        system_message = f"{safety_text}\n\n## Input Isolation\n\n{spotlight_instruction}"

        return PromptBundle(
            system_message=system_message,
            user_message=spotlighted_user,
            open_tag=open_tag,
            close_tag=close_tag,
        )

    # ------------------------------------------------------------------
    # Input inspection
    # ------------------------------------------------------------------

    def inspect_input(self, text: str) -> InputResult:
        """
        Full input inspection pipeline.

        1. Deobfuscate (Base64 / hex / URL decoding).
        2. Rule engine scan.
        3. Semantic classification.
        4. Compute combined risk score.
        5. Check for immediate-block conditions (high-severity rule match or
           high-confidence classifier result) — blocks before session state is
           consulted so obvious one-shot jailbreaks are caught instantly.
        6. Update cumulative session score (always, so repeat offenders
           accumulate even when some requests were immediately blocked).
        7. Apply cumulative session threshold if not already immediately blocked.
        8. Return InputResult with a reason that clearly identifies the block type.
        """
        # 1. Deobfuscate
        decoded = self._deobfuscator.decode(text)

        # 2. Rule scan
        rule_result = self._rule_engine.scan(decoded)

        # 3. Semantic classification
        classifier_result = self._classifier.classify(decoded)

        # 4. Combined risk score (weighted average)
        combined = rule_result.risk_score * 0.6 + classifier_result.confidence * 0.4

        # 5. Immediate-block checks (high-severity rule or high-confidence classifier)
        immediate_rule_block = (
            rule_result.risk_score >= self._immediate_block_rule_severity
        )
        immediate_classifier_block = (
            classifier_result.is_injection
            and classifier_result.confidence >= self._immediate_block_classifier_confidence
        )
        immediate_block = immediate_rule_block or immediate_classifier_block

        # 6. Always update cumulative score so the session history remains accurate
        cumulative = self._state.update(self.session_id, combined)

        # 7. Cumulative session threshold (catches repeat borderline attempts)
        session_blocked = self._state.is_flagged(self.session_id)

        blocked = immediate_block or session_blocked
        reason = _build_reason(
            blocked,
            immediate_rule_block,
            immediate_classifier_block,
            rule_result,
            classifier_result,
            cumulative,
        )

        return InputResult(
            blocked=blocked,
            risk_score=combined,
            cumulative_score=cumulative,
            decoded_text=decoded,
            rule_result=rule_result,
            classifier_result=classifier_result,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # RAG document inspection
    # ------------------------------------------------------------------

    def inspect_rag_document(self, document: str) -> tuple[str, RAGScanResult]:
        """
        Chunk, scan, and spotlight a retrieved RAG document.

        Returns ``(spotlighted_text, scan_result)``.  The
        ``scan_result.open_tag`` and ``scan_result.close_tag`` fields hold
        the randomized delimiter tags; embed them in your system message
        via :meth:`~promptshield.spotlighting.Spotlighter.build_spotlight_instruction`.
        """
        return self._rag.process_document(document, self._rule_engine)

    # ------------------------------------------------------------------
    # Response inspection
    # ------------------------------------------------------------------

    def inspect_response(self, response: str) -> tuple[str, ScanResult]:
        """
        Scan the LLM response for PII / sensitive data.

        If ``redact_responses`` is True, returns the redacted response.
        Otherwise returns the original text alongside the scan result.
        """
        scan_result = self._response_scanner.scan(response)
        if self.redact_responses and not scan_result.clean:
            return self._response_scanner.redact(response), scan_result
        return response, scan_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_reason(
    blocked: bool,
    immediate_rule_block: bool,
    immediate_classifier_block: bool,
    rule_result: RuleResult,
    classifier_result: ClassifierResult,
    cumulative: float,
) -> str:
    """Return a human-readable reason string that identifies the block type.

    Priority order:
    1. Immediate rule-based block (high-severity pattern match).
    2. Immediate classifier-based block (high-confidence injection label).
    3. Cumulative session block (repeated borderline attempts).
    4. Not blocked — may still describe rule/classifier signals.
    5. Clean (no signals at all).
    """
    if immediate_rule_block:
        return (
            f"immediate block: rule engine matched {len(rule_result.patterns)} "
            f"high-severity pattern(s) (score={rule_result.risk_score:.2f})"
        )
    if immediate_classifier_block:
        return (
            f"immediate block: classifier confidence {classifier_result.confidence:.2f} "
            f"exceeded threshold ({classifier_result.method})"
        )
    parts: list[str] = []
    if rule_result.matched:
        parts.append(f"rule match ({len(rule_result.patterns)} pattern(s))")
    if classifier_result.is_injection:
        parts.append(f"classifier flagged ({classifier_result.method}, confidence={classifier_result.confidence:.2f})")
    if blocked:
        parts.append(f"cumulative score {cumulative:.2f} exceeded threshold")
    if not parts:
        return "clean"
    return "; ".join(parts)
