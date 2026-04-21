"""
PromptShield — hybrid, stateful firewall for LLM security.
"""
from promptshield.firewall import PromptShield, InputResult, PromptBundle, TIER_MODELS
from promptshield.rule_engine import RuleResult
from promptshield.semantic_classifier import ClassifierResult, FREE_TIER_MODEL
from promptshield.ml_classifier import MLClassifier, PREMIUM_TIER_MODEL
from promptshield.rag_firewall import RAGScanResult
from promptshield.response_scanner import ScanResult, Finding
from promptshield.safety_system_message import SafetySystemMessage
from promptshield.spotlighting import Spotlighter

__all__ = [
    "PromptShield",
    "InputResult",
    "PromptBundle",
    "TIER_MODELS",
    "FREE_TIER_MODEL",
    "PREMIUM_TIER_MODEL",
    "MLClassifier",
    "RuleResult",
    "ClassifierResult",
    "RAGScanResult",
    "ScanResult",
    "Finding",
    "SafetySystemMessage",
    "Spotlighter",
]

__version__ = "0.1.0"
