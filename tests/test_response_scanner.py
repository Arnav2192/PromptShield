"""Tests for ResponseScanner."""
import pytest
from promptshield.response_scanner import ResponseScanner, ScanResult, Finding, _luhn_valid


@pytest.fixture
def scanner():
    return ResponseScanner()


# ---------------------------------------------------------------------------
# Luhn
# ---------------------------------------------------------------------------

def test_luhn_valid_visa():
    assert _luhn_valid("4532015112830366")


def test_luhn_invalid():
    assert not _luhn_valid("1234567890123456")


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------

def test_detects_email(scanner):
    result = scanner.scan("Contact us at admin@example.com for help.")
    types = [f.type for f in result.findings]
    assert "email" in types


def test_detects_phone(scanner):
    result = scanner.scan("Call me at 555-867-5309.")
    types = [f.type for f in result.findings]
    assert "phone" in types


def test_detects_ssn(scanner):
    result = scanner.scan("SSN: 123-45-6789")
    types = [f.type for f in result.findings]
    assert "ssn" in types


def test_detects_credit_card(scanner):
    # Valid Luhn: 4532015112830366
    result = scanner.scan("Card: 4532015112830366")
    types = [f.type for f in result.findings]
    assert "credit_card" in types


def test_invalid_credit_card_not_detected(scanner):
    result = scanner.scan("Number: 1234567890123456")
    types = [f.type for f in result.findings]
    assert "credit_card" not in types


def test_detects_api_key(scanner):
    result = scanner.scan("key = sk-abcdefghijklmnopqrstuvwxyz1234567890")
    types = [f.type for f in result.findings]
    assert "api_key" in types


def test_detects_password_kv(scanner):
    result = scanner.scan("password=mysecretpassword123")
    types = [f.type for f in result.findings]
    assert "password_kv" in types


def test_clean_text(scanner):
    result = scanner.scan("The weather today is sunny and warm.")
    assert result.clean
    assert result.findings == []


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def test_redact_email(scanner):
    redacted = scanner.redact("Email: user@example.com here.")
    assert "[REDACTED:email]" in redacted
    assert "user@example.com" not in redacted


def test_redact_multiple(scanner):
    text = "Email: a@b.com, SSN: 123-45-6789"
    redacted = scanner.redact(text)
    assert "[REDACTED:" in redacted


def test_redact_clean_text_unchanged(scanner):
    text = "Nothing sensitive here."
    assert scanner.redact(text) == text


def test_scan_result_type(scanner):
    result = scanner.scan("hello")
    assert isinstance(result, ScanResult)
    assert isinstance(result.findings, list)
