"""Tests for SafetySystemMessage."""
from __future__ import annotations
import pytest
from promptshield.safety_system_message import SafetySystemMessage


@pytest.fixture
def msg():
    return SafetySystemMessage()


def test_build_returns_string(msg):
    result = msg.build()
    assert isinstance(result, str)
    assert len(result) > 100


def test_contains_harmful_content_categories(msg):
    text = msg.build()
    assert "Hate Speech" in text
    assert "Violence" in text
    assert "Sexual Content" in text
    assert "Self-Harm" in text


def test_contains_copyright_policy(msg):
    text = msg.build()
    assert "copyright" in text.lower() or "intellectual property" in text.lower()


def test_contains_override_protection(msg):
    text = msg.build()
    assert "override" in text.lower() or "Override" in text


def test_contains_priority_statement(msg):
    text = msg.build()
    assert "highest priority" in text.lower() or "highest-priority" in text.lower()


def test_no_copyright_section_when_disabled():
    m = SafetySystemMessage(include_copyright_policy=False)
    text = m.build()
    # Harmful content still present
    assert "Hate Speech" in text
    # Copyright policy section omitted
    assert "Intellectual Property Policy" not in text


def test_additional_instructions_appended():
    m = SafetySystemMessage(additional_instructions="Always reply in English.")
    text = m.build()
    assert "Always reply in English." in text
    # Safety rules still present and come first
    assert text.index("Content Safety Policy") < text.index("Always reply in English.")


def test_contains_harmful_refusal_method(msg):
    assert msg.contains_harmful_refusal() is True


def test_contains_copyright_policy_method():
    m_with = SafetySystemMessage(include_copyright_policy=True)
    m_without = SafetySystemMessage(include_copyright_policy=False)
    assert m_with.contains_copyright_policy() is True
    assert m_without.contains_copyright_policy() is False


def test_prohibited_request_handling_instructions(msg):
    text = msg.build()
    # Must instruct the model to decline prohibited requests
    assert "decline" in text.lower() or "refuse" in text.lower() or "Decline" in text


def test_refusal_of_roleplay_bypass(msg):
    text = msg.build()
    # Must mention that roleplay / fictional framing cannot bypass rules
    assert "hypothetical" in text.lower() or "fictional" in text.lower() or "pretend" in text.lower()
