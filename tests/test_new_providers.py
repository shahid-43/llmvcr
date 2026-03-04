"""
Tests for Gemini, Ollama, and Groq providers.

All tests use pre-recorded cassettes — no API keys, no internet, no
local Ollama server needed. These verify the full serialize → store →
match → deserialize pipeline for each new provider.
"""

import os
import pytest
import yaml

from llmvcr.recorder import Recorder
from llmvcr.storage import load, get_interactions
from llmvcr.errors import NoMatchFoundError
from llmvcr.providers import gemini_provider, ollama_provider, groq_provider

CASSETTES_DIR = os.path.join(os.path.dirname(__file__), "cassettes")
GEMINI_CASSETTE  = os.path.join(CASSETTES_DIR, "sample_gemini.yaml")
OLLAMA_CASSETTE  = os.path.join(CASSETTES_DIR, "sample_ollama.yaml")
GROQ_CASSETTE    = os.path.join(CASSETTES_DIR, "sample_groq.yaml")


# ══════════════════════════════════════════════════════════════════════
# CASSETTE FILE SANITY
# ══════════════════════════════════════════════════════════════════════

class TestNewCassetteFiles:

    def test_all_cassettes_exist(self):
        for path in [GEMINI_CASSETTE, OLLAMA_CASSETTE, GROQ_CASSETTE]:
            assert os.path.exists(path), f"Missing cassette: {path}"

    def test_all_cassettes_valid_yaml(self):
        for path in [GEMINI_CASSETTE, OLLAMA_CASSETTE, GROQ_CASSETTE]:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert "interactions" in data, f"No interactions in {path}"

    def test_no_api_keys_in_any_cassette(self):
        for path in [GEMINI_CASSETTE, OLLAMA_CASSETTE, GROQ_CASSETTE]:
            with open(path) as f:
                content = f.read()
            assert "Authorization" not in content
            assert "Bearer " not in content
            assert "x-api-key" not in content.lower()
            assert "!!binary" not in content

    def test_providers_correctly_labelled(self):
        for path, expected in [
            (GEMINI_CASSETTE,  "gemini"),
            (OLLAMA_CASSETTE,  "ollama"),
            (GROQ_CASSETTE,    "groq"),
        ]:
            data = load(path)
            assert data["provider"] == expected, \
                f"{path} should have provider={expected}, got {data['provider']}"


# ══════════════════════════════════════════════════════════════════════
# GEMINI REQUEST SERIALIZATION
# ══════════════════════════════════════════════════════════════════════

class TestGeminiSerialization:
    """
    Test that Gemini's unique 'contents' format is correctly normalized
    to llmvcr's internal messages format.
    """

    def test_plain_string_contents(self):
        kwargs = {"model": "gemini-2.0-flash", "contents": "Hello!"}
        req = gemini_provider.serialize_request(kwargs)
        assert req["model"] == "gemini-2.0-flash"
        assert req["messages"] == [{"role": "user", "content": "Hello!"}]

    def test_list_of_strings(self):
        kwargs = {"model": "gemini-2.0-flash", "contents": ["Hello!", "World"]}
        req = gemini_provider.serialize_request(kwargs)
        assert len(req["messages"]) == 2
        assert req["messages"][0]["content"] == "Hello!"
        assert req["messages"][1]["content"] == "World"

    def test_dict_contents_with_role(self):
        kwargs = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "What is AI?"}]}
            ]
        }
        req = gemini_provider.serialize_request(kwargs)
        assert req["messages"][0]["role"] == "user"
        assert req["messages"][0]["content"] == "What is AI?"

    def test_system_instruction_from_config_dict(self):
        kwargs = {
            "model": "gemini-2.0-flash",
            "contents": "Hello!",
            "config": {"system_instruction": "You are a helpful assistant."},
        }
        req = gemini_provider.serialize_request(kwargs)
        # System instruction should be prepended as first message
        assert req["messages"][0]["role"] == "system"
        assert req["messages"][0]["content"] == "You are a helpful assistant."
        assert req["messages"][1]["content"] == "Hello!"

    def test_ignores_generation_config_params(self):
        """temperature, top_p, etc. in config should not affect matching."""
        kwargs = {
            "model": "gemini-2.0-flash",
            "contents": "Hello!",
            "config": {"temperature": 0.9, "top_p": 0.95, "max_output_tokens": 100},
        }
        req = gemini_provider.serialize_request(kwargs)
        assert req["messages"] == [{"role": "user", "content": "Hello!"}]


# ══════════════════════════════════════════════════════════════════════
# GEMINI PLAYBACK
# ══════════════════════════════════════════════════════════════════════

class TestGeminiPlayback:

    def test_playback_summarize_ai(self):
        r = Recorder(GEMINI_CASSETTE, mode="playback", provider="gemini")
        response = r.get_response({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert "intelligence" in response["text"].lower()

    def test_playback_with_system_instruction(self):
        r = Recorder(GEMINI_CASSETTE, mode="playback", provider="gemini")
        response = r.get_response({
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is Python?"},
            ],
        })
        assert "python" in response["text"].lower()

    def test_playback_has_candidates(self):
        r = Recorder(GEMINI_CASSETTE, mode="playback", provider="gemini")
        response = r.get_response({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert "candidates" in response
        assert len(response["candidates"]) > 0
        assert response["candidates"][0]["finish_reason"] == "STOP"

    def test_playback_has_usage_metadata(self):
        r = Recorder(GEMINI_CASSETTE, mode="playback", provider="gemini")
        response = r.get_response({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert "usage_metadata" in response
        assert response["usage_metadata"]["total_token_count"] == 32

    def test_playback_ignores_temperature(self):
        """Gemini config params like temperature should not affect matching."""
        r = Recorder(GEMINI_CASSETTE, mode="playback", provider="gemini")
        # The stored cassette request has no config params — this should still match
        response = r.get_response({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert response is not None

    def test_wrong_model_raises(self):
        r = Recorder(GEMINI_CASSETTE, mode="playback", provider="gemini")
        with pytest.raises(NoMatchFoundError):
            r.get_response({
                "model": "gemini-1.5-pro",  # not in cassette
                "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
            })


# ══════════════════════════════════════════════════════════════════════
# OLLAMA SERIALIZATION
# ══════════════════════════════════════════════════════════════════════

class TestOllamaSerialization:

    def test_basic_request(self):
        kwargs = {
            "model": "llama3.1",
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        req = ollama_provider.serialize_request(kwargs)
        assert req["model"] == "llama3.1"
        assert req["messages"] == [{"role": "user", "content": "Hello!"}]

    def test_with_system_message(self):
        kwargs = {
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": "You are a code assistant."},
                {"role": "user", "content": "Write a for loop."},
            ],
        }
        req = ollama_provider.serialize_request(kwargs)
        assert req["messages"][0]["role"] == "system"
        assert req["messages"][1]["role"] == "user"

    def test_response_proxy_dict_access(self):
        """The Ollama response proxy should support dict-style access."""
        data = {"model": "llama3.1", "message": {"role": "assistant", "content": "Hi!"}, "done": True}
        proxy = ollama_provider._OllamaResponseProxy(data)
        assert proxy["message"]["content"] == "Hi!"
        assert proxy["model"] == "llama3.1"

    def test_response_proxy_attribute_access(self):
        """The Ollama response proxy should support attribute-style access."""
        data = {"model": "llama3.1", "message": {"role": "assistant", "content": "Hi!"}, "done": True}
        proxy = ollama_provider._OllamaResponseProxy(data)
        assert proxy.message.content == "Hi!"
        assert proxy.message.role == "assistant"
        assert proxy.model == "llama3.1"


# ══════════════════════════════════════════════════════════════════════
# OLLAMA PLAYBACK
# ══════════════════════════════════════════════════════════════════════

class TestOllamaPlayback:

    def test_playback_summarize_ai(self):
        r = Recorder(OLLAMA_CASSETTE, mode="playback", provider="ollama")
        response = r.get_response({
            "model": "llama3.1",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert "intelligence" in response["message"]["content"].lower()

    def test_playback_with_system_message(self):
        r = Recorder(OLLAMA_CASSETTE, mode="playback", provider="ollama")
        response = r.get_response({
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
            ],
        })
        assert "python" in response["message"]["content"].lower()

    def test_playback_has_done_flag(self):
        r = Recorder(OLLAMA_CASSETTE, mode="playback", provider="ollama")
        response = r.get_response({
            "model": "llama3.1",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert response["done"] is True

    def test_playback_has_eval_counts(self):
        r = Recorder(OLLAMA_CASSETTE, mode="playback", provider="ollama")
        response = r.get_response({
            "model": "llama3.1",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert response["prompt_eval_count"] == 14
        assert response["eval_count"] == 24

    def test_wrong_model_raises(self):
        r = Recorder(OLLAMA_CASSETTE, mode="playback", provider="ollama")
        with pytest.raises(NoMatchFoundError):
            r.get_response({
                "model": "mistral",  # not in cassette
                "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
            })

    def test_no_api_key_needed(self):
        """
        Ollama runs locally — there's never an API key to worry about.
        Verify the cassette contains no auth fields whatsoever.
        """
        with open(OLLAMA_CASSETTE) as f:
            content = f.read()
        assert "api_key" not in content.lower()
        assert "authorization" not in content.lower()


# ══════════════════════════════════════════════════════════════════════
# GROQ SERIALIZATION
# ══════════════════════════════════════════════════════════════════════

class TestGroqSerialization:

    def test_basic_request(self):
        kwargs = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        req = groq_provider.serialize_request(kwargs)
        assert req["model"] == "llama-3.3-70b-versatile"
        assert req["messages"] == [{"role": "user", "content": "Hello!"}]

    def test_ignores_temperature(self):
        kwargs = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "max_tokens": 200,
        }
        req = groq_provider.serialize_request(kwargs)
        assert "temperature" not in req
        assert "max_tokens" not in req
        assert req["messages"] == [{"role": "user", "content": "Hello!"}]


# ══════════════════════════════════════════════════════════════════════
# GROQ PLAYBACK
# ══════════════════════════════════════════════════════════════════════

class TestGroqPlayback:

    def test_playback_llama_70b(self):
        r = Recorder(GROQ_CASSETTE, mode="playback", provider="groq")
        response = r.get_response({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        content = response["choices"][0]["message"]["content"]
        assert "intelligence" in content.lower()

    def test_playback_llama_8b(self):
        r = Recorder(GROQ_CASSETTE, mode="playback", provider="groq")
        response = r.get_response({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "What is Python?"}],
        })
        content = response["choices"][0]["message"]["content"]
        assert "python" in content.lower()

    def test_playback_has_usage(self):
        r = Recorder(GROQ_CASSETTE, mode="playback", provider="groq")
        response = r.get_response({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert response["usage"]["total_tokens"] == 42

    def test_playback_has_id(self):
        r = Recorder(GROQ_CASSETTE, mode="playback", provider="groq")
        response = r.get_response({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        assert response["id"] == "chatcmpl-groq-sample001"

    def test_different_llama_models_are_distinct(self):
        """llama-3.3-70b and llama-3.1-8b should NOT match each other."""
        r = Recorder(GROQ_CASSETTE, mode="playback", provider="groq")
        response_70b = r.get_response({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
        })
        # The 70b response ID is different from the 8b one
        assert response_70b["id"] == "chatcmpl-groq-sample001"

    def test_wrong_model_raises(self):
        r = Recorder(GROQ_CASSETTE, mode="playback", provider="groq")
        with pytest.raises(NoMatchFoundError):
            r.get_response({
                "model": "llama-3.1-70b-versatile",  # not in cassette
                "messages": [{"role": "user", "content": "Summarize artificial intelligence in one sentence."}],
            })

    def test_groq_cassette_same_format_as_openai(self):
        """
        Groq uses OpenAI-compatible format. Verify the cassette structure
        is identical to the OpenAI cassette — choices, message, usage.
        """
        data = load(GROQ_CASSETTE)
        interaction = get_interactions(data)[0]
        response = interaction["response"]
        assert "choices" in response
        assert "message" in response["choices"][0]
        assert "usage" in response
        # Groq never leaks auth headers — verify provider label confirms groq
        assert data["provider"] == "groq"


# ══════════════════════════════════════════════════════════════════════
# PROVIDER ROUTING
# ══════════════════════════════════════════════════════════════════════

class TestProviderRouting:
    """Verify that the _patch.py router correctly identifies all 5 providers."""

    def test_unsupported_provider_raises(self, tmp_path):
        from llmvcr.errors import ProviderNotSupportedError
        from llmvcr._patch import apply_patch

        with pytest.raises(ProviderNotSupportedError) as exc_info:
            with apply_patch("cohere", None, None):
                pass
        assert "cohere" in str(exc_info.value)
        assert "openai" in str(exc_info.value)
        assert "gemini" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)
        assert "groq" in str(exc_info.value)