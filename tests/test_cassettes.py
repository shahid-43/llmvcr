"""
Integration tests using the sample cassettes in tests/cassettes/.

These tests verify the full record → playback pipeline without ever
touching a real LLM API. They use pre-recorded cassettes committed
to the repo, so they run instantly, free, and offline.

This is exactly the workflow llmvcr enables for your own projects.
"""

import os
import pytest
import yaml

from llmvcr.recorder import Recorder
from llmvcr.storage import load, get_interactions
from llmvcr.matching import find_match
from llmvcr.errors import CassetteNotFoundError, NoMatchFoundError

CASSETTES_DIR = os.path.join(os.path.dirname(__file__), "cassettes")
OPENAI_CASSETTE = os.path.join(CASSETTES_DIR, "sample_openai.yaml")
ANTHROPIC_CASSETTE = os.path.join(CASSETTES_DIR, "sample_anthropic.yaml")


# ── Cassette file sanity checks ───────────────────────────────────────────────

class TestCassetteFiles:
    """Verify the cassette files themselves are well-formed and safe."""

    def test_openai_cassette_exists(self):
        assert os.path.exists(OPENAI_CASSETTE), \
            f"Sample cassette missing: {OPENAI_CASSETTE}"

    def test_anthropic_cassette_exists(self):
        assert os.path.exists(ANTHROPIC_CASSETTE), \
            f"Sample cassette missing: {ANTHROPIC_CASSETTE}"

    def test_openai_cassette_is_valid_yaml(self):
        with open(OPENAI_CASSETTE) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert "interactions" in data

    def test_cassettes_have_no_api_keys(self):
        """Security check: no credentials should ever appear in a cassette."""
        for path in [OPENAI_CASSETTE, ANTHROPIC_CASSETTE]:
            with open(path) as f:
                content = f.read()
            assert "Authorization" not in content, f"Auth header found in {path}"
            assert "Bearer sk-" not in content, f"API key found in {path}"
            assert "x-api-key" not in content.lower(), f"API key found in {path}"
            assert "!!binary" not in content, f"Binary blob found in {path}"

    def test_cassettes_are_human_readable(self):
        """All response content should be visible as plain text."""
        with open(OPENAI_CASSETTE) as f:
            content = f.read()
        assert "Artificial intelligence" in content
        assert "gpt-4" in content

    def test_openai_cassette_has_three_interactions(self):
        data = load(OPENAI_CASSETTE)
        interactions = get_interactions(data)
        assert len(interactions) == 3

    def test_anthropic_cassette_has_two_interactions(self):
        data = load(ANTHROPIC_CASSETTE)
        interactions = get_interactions(data)
        assert len(interactions) == 2


# ── OpenAI playback via Recorder ─────────────────────────────────────────────

class TestOpenAIPlayback:
    """Test replaying the OpenAI sample cassette through the Recorder."""

    def test_playback_summarize_ai(self):
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        response = r.get_response({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        content = response["choices"][0]["message"]["content"]
        assert "intelligence" in content.lower()
        assert "artificial" in content.lower()

    def test_playback_what_is_python(self):
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        response = r.get_response({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is Python?"}],
        })
        content = response["choices"][0]["message"]["content"]
        assert "python" in content.lower()

    def test_playback_with_system_message(self):
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        response = r.get_response({
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"},
            ],
        })
        content = response["choices"][0]["message"]["content"]
        assert "hello" in content.lower()

    def test_playback_ignores_temperature(self):
        """The key llmvcr advantage: extra params don't break playback."""
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        response = r.get_response({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "temperature": 0.9,    # ← would break VCR.py
            "max_tokens": 500,     # ← would break VCR.py
            "top_p": 0.95,         # ← would break VCR.py
        })
        assert response is not None
        assert response["choices"][0]["message"]["content"]

    def test_playback_response_has_usage(self):
        """Usage/token counts should be preserved in the cassette."""
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        response = r.get_response({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert "usage" in response
        assert response["usage"]["total_tokens"] == 36

    def test_playback_response_has_id(self):
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        response = r.get_response({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is Python?"}],
        })
        assert response["id"] == "chatcmpl-sample002"

    def test_playback_wrong_prompt_raises(self):
        """A prompt not in the cassette should raise NoMatchFoundError."""
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        with pytest.raises(NoMatchFoundError) as exc_info:
            r.get_response({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "This prompt is not in the cassette"}],
            })
        assert "sample_openai.yaml" in str(exc_info.value)

    def test_playback_wrong_model_raises(self):
        """Same prompt but different model should not match."""
        r = Recorder(OPENAI_CASSETTE, mode="playback", provider="openai")
        with pytest.raises(NoMatchFoundError):
            r.get_response({
                "model": "gpt-4o",   # different model — not in cassette
                "messages": [{"role": "user", "content": "What is Python?"}],
            })


# ── Anthropic playback via Recorder ──────────────────────────────────────────

class TestAnthropicPlayback:
    """Test replaying the Anthropic sample cassette through the Recorder."""

    def test_playback_summarize_ai(self):
        r = Recorder(ANTHROPIC_CASSETTE, mode="playback", provider="anthropic")
        response = r.get_response({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Summarize artificial intelligence in one sentence."},
            ],
        })
        content = response["content"][0]["text"]
        assert "intelligence" in content.lower()

    def test_playback_what_is_python(self):
        r = Recorder(ANTHROPIC_CASSETTE, mode="playback", provider="anthropic")
        response = r.get_response({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "What is Python?"}],
        })
        content = response["content"][0]["text"]
        assert "python" in content.lower()

    def test_anthropic_response_has_usage(self):
        r = Recorder(ANTHROPIC_CASSETTE, mode="playback", provider="anthropic")
        response = r.get_response({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "What is Python?"}],
        })
        assert "usage" in response
        assert response["usage"]["input_tokens"] == 12


# ── Full record → playback round trip ────────────────────────────────────────

class TestRoundTrip:
    """
    Simulate the real user workflow:
      1. Record (fake the API call, inject a response)
      2. Save cassette to disk
      3. Load and replay — verify we get the same response back
    """

    def test_record_then_playback(self, tmp_path):
        path = str(tmp_path / "round_trip.yaml")

        fake_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Round trip test"}],
        }
        fake_response = {
            "id": "chatcmpl-roundtrip",
            "choices": [{"message": {"role": "assistant", "content": "Round trip success!"}}],
            "usage": {"total_tokens": 10},
        }

        # Step 1: Record
        recorder = Recorder(path, mode="record", provider="openai")
        recorder.on_response(fake_request, fake_response)
        recorder.save()

        # Step 2: Playback
        replayer = Recorder(path, mode="playback", provider="openai")
        response = replayer.get_response(fake_request)

        assert response["id"] == "chatcmpl-roundtrip"
        assert response["choices"][0]["message"]["content"] == "Round trip success!"

    def test_auto_mode_full_cycle(self, tmp_path):
        """Auto mode: first call records, second call replays."""
        path = str(tmp_path / "auto.yaml")

        fake_request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Auto test"}]}
        fake_response = {"id": "auto-001", "choices": [{"message": {"content": "Auto response"}}]}

        # First run — no cassette exists, should record
        r1 = Recorder(path, mode="auto", provider="openai")
        assert r1.should_record() is True
        r1.on_response(fake_request, fake_response)
        r1.save()

        # Second run — cassette now exists, should replay
        r2 = Recorder(path, mode="auto", provider="openai")
        assert r2.should_record() is False
        response = r2.get_response(fake_request)
        assert response["id"] == "auto-001"

    def test_multiple_interactions_round_trip(self, tmp_path):
        """A cassette with multiple interactions replays them correctly."""
        path = str(tmp_path / "multi.yaml")

        interactions = [
            ({"model": "gpt-4", "messages": [{"role": "user", "content": f"Question {i}"}]},
             {"id": f"cmpl-{i}", "choices": [{"message": {"content": f"Answer {i}"}}]})
            for i in range(4)
        ]

        recorder = Recorder(path, mode="record", provider="openai")
        for req, resp in interactions:
            recorder.on_response(req, resp)
        recorder.save()

        replayer = Recorder(path, mode="playback", provider="openai")
        for i, (req, _) in enumerate(interactions):
            response = replayer.get_response(req)
            assert response["id"] == f"cmpl-{i}"
            assert response["choices"][0]["message"]["content"] == f"Answer {i}"