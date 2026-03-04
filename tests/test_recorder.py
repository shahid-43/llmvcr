"""
Tests for the Recorder engine.
No LLM API keys needed — we test the record/playback logic directly.
"""

import pytest
from llmvcr.recorder import Recorder
from llmvcr.errors import CassetteNotFoundError, NoMatchFoundError


SAMPLE_REQUEST = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
}
SAMPLE_RESPONSE = {
    "id": "chatcmpl-test",
    "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
    "usage": {"total_tokens": 9},
}


def test_record_mode_should_record(tmp_path):
    """In record mode, should_record() returns True."""
    r = Recorder(str(tmp_path / "c.yaml"), mode="record", provider="openai")
    assert r.should_record() is True


def test_playback_mode_raises_if_missing(tmp_path):
    """In playback mode, raise CassetteNotFoundError if no cassette on disk."""
    with pytest.raises(CassetteNotFoundError) as exc_info:
        Recorder(str(tmp_path / "missing.yaml"), mode="playback", provider="openai")
    assert "missing.yaml" in str(exc_info.value)


def test_auto_mode_records_when_no_cassette(tmp_path):
    """In auto mode with no cassette, should_record() returns True."""
    r = Recorder(str(tmp_path / "c.yaml"), mode="auto", provider="openai")
    assert r.should_record() is True


def test_auto_mode_replays_when_cassette_exists(tmp_path):
    """In auto mode with an existing cassette, should_record() returns False."""
    # First create a cassette
    r1 = Recorder(str(tmp_path / "c.yaml"), mode="record", provider="openai")
    r1.on_response(SAMPLE_REQUEST, SAMPLE_RESPONSE)
    r1.save()

    # Now load it in auto mode
    r2 = Recorder(str(tmp_path / "c.yaml"), mode="auto", provider="openai")
    assert r2.should_record() is False


def test_record_then_retrieve(tmp_path):
    """Recording a response and then retrieving it should return the same data."""
    path = str(tmp_path / "c.yaml")
    r = Recorder(path, mode="record", provider="openai")
    r.on_response(SAMPLE_REQUEST, SAMPLE_RESPONSE)
    r.save()

    # Load it back in playback mode
    r2 = Recorder(path, mode="playback", provider="openai")
    response = r2.get_response(SAMPLE_REQUEST)
    assert response["id"] == "chatcmpl-test"
    assert response["choices"][0]["message"]["content"] == "Hi there!"


def test_no_match_raises(tmp_path):
    """get_response() raises NoMatchFoundError if no matching interaction."""
    path = str(tmp_path / "c.yaml")
    r = Recorder(path, mode="record", provider="openai")
    r.on_response(SAMPLE_REQUEST, SAMPLE_RESPONSE)
    r.save()

    r2 = Recorder(path, mode="playback", provider="openai")
    with pytest.raises(NoMatchFoundError):
        r2.get_response({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "A completely different question"}],
        })


def test_make_callbacks_record_mode(tmp_path):
    """In record mode, record_cb is set and playback_cb is None."""
    r = Recorder(str(tmp_path / "c.yaml"), mode="record", provider="openai")
    record_cb, playback_cb = r.make_callbacks()
    assert record_cb is not None
    assert playback_cb is None


def test_make_callbacks_playback_mode(tmp_path):
    """In playback mode, playback_cb is set and record_cb is None."""
    path = str(tmp_path / "c.yaml")
    r1 = Recorder(path, mode="record", provider="openai")
    r1.on_response(SAMPLE_REQUEST, SAMPLE_RESPONSE)
    r1.save()

    r2 = Recorder(path, mode="playback", provider="openai")
    record_cb, playback_cb = r2.make_callbacks()
    assert record_cb is None
    assert playback_cb is not None


def test_not_dirty_no_write(tmp_path):
    """save() should not write a file if nothing was recorded."""
    path = str(tmp_path / "c.yaml")
    r = Recorder(path, mode="record", provider="openai")
    r.save()  # Nothing recorded
    import os
    assert not os.path.exists(path)