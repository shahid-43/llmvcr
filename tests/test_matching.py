"""
Tests for the semantic request matching module.
These tests do NOT require any LLM API keys — they test pure logic.
"""

import pytest
from llmvcr.matching import find_match, request_summary


SAMPLE_INTERACTIONS = [
    {
        "request": {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Summarize AI"}],
        },
        "response": {"choices": [{"message": {"content": "AI is..."}}]},
    },
    {
        "request": {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is Python?"}],
        },
        "response": {"choices": [{"message": {"content": "Python is..."}}]},
    },
]


def test_exact_match():
    """A request identical to a stored one should match."""
    request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Summarize AI"}],
    }
    result = find_match(request, SAMPLE_INTERACTIONS)
    assert result is not None
    assert result["response"]["choices"][0]["message"]["content"] == "AI is..."


def test_match_ignores_temperature():
    """Adding temperature to a request should not break the match."""
    request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Summarize AI"}],
        "temperature": 0.7,   # ← this would break VCR.py, not llmvcr
        "max_tokens": 100,
    }
    result = find_match(request, SAMPLE_INTERACTIONS)
    assert result is not None, "Should still match despite extra parameters"


def test_match_ignores_max_tokens():
    """Adding max_tokens should not break the match."""
    request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Summarize AI"}],
        "max_tokens": 500,
    }
    result = find_match(request, SAMPLE_INTERACTIONS)
    assert result is not None


def test_no_match_different_content():
    """A request with different message content should not match."""
    request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "A completely different question"}],
    }
    result = find_match(request, SAMPLE_INTERACTIONS)
    assert result is None


def test_no_match_different_model():
    """A request with a different model should not match."""
    request = {
        "model": "gpt-3.5-turbo",  # different model
        "messages": [{"role": "user", "content": "Summarize AI"}],
    }
    result = find_match(request, SAMPLE_INTERACTIONS)
    assert result is None


def test_match_correct_interaction():
    """The correct interaction is returned when multiple are stored."""
    request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "What is Python?"}],
    }
    result = find_match(request, SAMPLE_INTERACTIONS)
    assert result is not None
    assert result["response"]["choices"][0]["message"]["content"] == "Python is..."


def test_empty_interactions():
    """No match in an empty cassette."""
    result = find_match({"model": "gpt-4", "messages": []}, [])
    assert result is None


def test_request_summary_truncates_long_content():
    """Long message content should be truncated in the summary."""
    request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "x" * 200}],
    }
    summary = request_summary(request)
    assert len(summary) < 250
    assert "..." in summary


def test_request_summary_shows_model():
    request = {"model": "claude-3", "messages": [{"role": "user", "content": "Hi"}]}
    summary = request_summary(request)
    assert "claude-3" in summary