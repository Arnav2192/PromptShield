"""Tests for the Spotlighter (Microsoft MSRC spotlighting techniques)."""
from __future__ import annotations
import base64
import pytest
from promptshield.spotlighting import Spotlighter


@pytest.fixture
def spotlighter():
    return Spotlighter()


# ---------------------------------------------------------------------------
# Technique 1 — Delimiting
# ---------------------------------------------------------------------------

def test_delimit_returns_three_values(spotlighter):
    result = spotlighter.delimit("hello world")
    assert len(result) == 3
    text, open_tag, close_tag = result
    assert isinstance(text, str)
    assert isinstance(open_tag, str)
    assert isinstance(close_tag, str)


def test_delimit_wraps_content(spotlighter):
    text, open_tag, close_tag = spotlighter.delimit("some content")
    assert open_tag in text
    assert close_tag in text
    assert "some content" in text


def test_delimit_tag_format(spotlighter):
    _, open_tag, close_tag = spotlighter.delimit("x")
    assert open_tag.startswith("<<")
    assert open_tag.endswith("_START>>")
    assert close_tag.startswith("<<")
    assert close_tag.endswith("_END>>")


def test_delimit_unique_tags_per_call(spotlighter):
    _, tag1, _ = spotlighter.delimit("call 1")
    _, tag2, _ = spotlighter.delimit("call 2")
    assert tag1 != tag2


def test_delimit_tag_length(spotlighter):
    _, open_tag, _ = spotlighter.delimit("x")
    # <<{12-char-id}_START>> = 2 + 12 + 7 = 21 chars
    inner = open_tag[2:-8]  # strip << and _START>>
    assert len(inner) == spotlighter.tag_length


def test_delimit_empty_string(spotlighter):
    text, open_tag, close_tag = spotlighter.delimit("")
    assert open_tag in text
    assert close_tag in text


# ---------------------------------------------------------------------------
# Technique 2 — Datamarking
# ---------------------------------------------------------------------------

def test_datamark_replaces_spaces(spotlighter):
    result = spotlighter.datamark("hello world")
    assert " " not in result
    assert spotlighter.datamark_char in result


def test_datamark_replaces_tabs(spotlighter):
    result = spotlighter.datamark("col1\tcol2")
    assert "\t" not in result
    assert spotlighter.datamark_char in result


def test_datamark_preserves_newlines(spotlighter):
    result = spotlighter.datamark("line1\nline2")
    assert "\n" in result


def test_datamark_default_char():
    s = Spotlighter()
    assert s.datamark_char == "^"
    assert "^" in s.datamark("a b")


def test_datamark_custom_char():
    s = Spotlighter(datamark_char="|")
    assert "|" in s.datamark("a b")


# ---------------------------------------------------------------------------
# Technique 3 — Encoding
# ---------------------------------------------------------------------------

def test_encode_produces_valid_base64(spotlighter):
    encoded = spotlighter.encode("hello world")
    decoded = base64.b64decode(encoded).decode("utf-8")
    assert decoded == "hello world"


def test_encode_disrupts_injection_phrase(spotlighter):
    phrase = "ignore previous instructions"
    encoded = spotlighter.encode(phrase)
    # Encoded form must not contain the raw phrase
    assert "ignore" not in encoded
    assert "instructions" not in encoded


def test_encode_unicode(spotlighter):
    text = "café au lait"
    encoded = spotlighter.encode(text)
    decoded = base64.b64decode(encoded).decode("utf-8")
    assert decoded == text


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def test_spotlight_user_input(spotlighter):
    result = spotlighter.spotlight_user_input("user text")
    assert len(result) == 3
    text, open_tag, close_tag = result
    assert "user text" in text
    assert open_tag in text


def test_spotlight_rag_document(spotlighter):
    result = spotlighter.spotlight_rag_document("doc text")
    assert len(result) == 3
    text, open_tag, close_tag = result
    assert "doc text" in text


def test_build_spotlight_instruction_contains_tags(spotlighter):
    _, open_tag, close_tag = spotlighter.delimit("something")
    instruction = spotlighter.build_spotlight_instruction(open_tag, close_tag)
    assert open_tag in instruction
    assert close_tag in instruction


def test_build_spotlight_instruction_contains_content_type(spotlighter):
    _, open_tag, close_tag = spotlighter.delimit("x")
    instruction = spotlighter.build_spotlight_instruction(
        open_tag, close_tag, content_type="retrieved document"
    )
    assert "retrieved document" in instruction


def test_build_spotlight_instruction_warns_against_following_instructions(spotlighter):
    _, open_tag, close_tag = spotlighter.delimit("x")
    instruction = spotlighter.build_spotlight_instruction(open_tag, close_tag)
    lower = instruction.lower()
    assert "do not follow" in lower or "not follow" in lower


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_tag_length_too_short_raises():
    with pytest.raises(ValueError):
        Spotlighter(tag_length=3)


def test_custom_tag_length():
    s = Spotlighter(tag_length=16)
    _, open_tag, _ = s.delimit("x")
    inner = open_tag[2:-8]  # strip << and _START>>
    assert len(inner) == 16
