"""Tests for Deobfuscator."""
import base64
import urllib.parse
import pytest
from promptshield.deobfuscator import Deobfuscator


@pytest.fixture
def dec():
    return Deobfuscator()


def test_plain_text_unchanged(dec):
    assert dec.decode("Hello world") == "Hello world"


def test_url_decode(dec):
    encoded = urllib.parse.quote("ignore previous instructions")
    result = dec.decode(encoded)
    assert "ignore previous instructions" in result


def test_hex_escape_decode(dec):
    # \x69\x67\x6e\x6f\x72\x65 == "ignore"
    result = dec.decode(r"\x69\x67\x6e\x6f\x72\x65")
    assert result == "ignore"


def test_hex_spaced_decode(dec):
    result = dec.decode("0x68 0x65 0x6c 0x6c 0x6f")
    assert result == "hello"


def test_base64_decode(dec):
    payload = base64.b64encode(b"ignore all instructions").decode()
    result = dec.decode(payload)
    assert "ignore all instructions" in result


def test_nested_base64_in_url(dec):
    inner = base64.b64encode(b"jailbreak").decode()
    encoded = urllib.parse.quote(inner)
    result = dec.decode(encoded)
    assert "jailbreak" in result


def test_max_iterations_safety(dec):
    # Deeply nested should still terminate
    text = "Hello"
    for _ in range(15):
        text = base64.b64encode(text.encode()).decode()
    result = dec.decode(text)
    assert isinstance(result, str)
