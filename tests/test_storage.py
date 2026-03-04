"""
Tests for cassette storage (read/write YAML).
No LLM API keys needed.
"""

import os
import pytest
import yaml
from llmvcr import storage


SAMPLE_INTERACTIONS = [
    {
        "request": {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        "response": {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        },
    }
]


def test_save_and_load(tmp_path):
    """Saving then loading a cassette should return identical interactions."""
    path = str(tmp_path / "test.yaml")
    storage.save(path, "openai", SAMPLE_INTERACTIONS)

    data = storage.load(path)
    assert data is not None
    assert data["provider"] == "openai"
    assert data["llmvcr_version"] == "0.1"
    assert len(data["interactions"]) == 1

    loaded = storage.get_interactions(data)
    assert loaded[0]["request"]["model"] == "gpt-4"
    assert loaded[0]["response"]["id"] == "chatcmpl-test"


def test_load_nonexistent_returns_none(tmp_path):
    """Loading a missing cassette returns None (does not raise)."""
    result = storage.load(str(tmp_path / "does_not_exist.yaml"))
    assert result is None


def test_save_creates_directories(tmp_path):
    """save() should create intermediate directories if they don't exist."""
    path = str(tmp_path / "nested" / "dir" / "test.yaml")
    storage.save(path, "openai", SAMPLE_INTERACTIONS)
    assert os.path.exists(path)


def test_cassette_is_human_readable(tmp_path):
    """The cassette YAML should be readable plain text — not binary blobs."""
    path = str(tmp_path / "test.yaml")
    storage.save(path, "openai", SAMPLE_INTERACTIONS)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Should be valid readable YAML with our content visible as plain text
    assert "gpt-4" in content
    assert "Hello!" in content
    assert "Hi there!" in content
    # Should NOT contain binary indicators
    assert "!!binary" not in content
    assert "base64" not in content.lower()


def test_cassette_has_no_auth_headers(tmp_path):
    """Cassettes must never contain API keys or auth tokens."""
    path = str(tmp_path / "test.yaml")
    storage.save(path, "openai", SAMPLE_INTERACTIONS)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # None of these should appear in a cassette
    assert "Authorization" not in content
    assert "Bearer sk-" not in content
    assert "x-api-key" not in content.lower()


def test_recorded_at_is_set(tmp_path):
    """Cassette should record the timestamp it was created."""
    path = str(tmp_path / "test.yaml")
    storage.save(path, "openai", SAMPLE_INTERACTIONS)

    data = storage.load(path)
    assert "recorded_at" in data
    assert data["recorded_at"]  # not empty


def test_save_multiple_interactions(tmp_path):
    """Multiple interactions should all be preserved."""
    interactions = [
        {"request": {"model": "gpt-4", "messages": [{"role": "user", "content": f"Question {i}"}]},
         "response": {"choices": [{"message": {"content": f"Answer {i}"}}]}}
        for i in range(5)
    ]
    path = str(tmp_path / "multi.yaml")
    storage.save(path, "openai", interactions)

    data = storage.load(path)
    loaded = storage.get_interactions(data)
    assert len(loaded) == 5
    assert loaded[3]["request"]["messages"][0]["content"] == "Question 3"