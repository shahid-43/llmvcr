"""
llmvcr — Record and replay LLM API responses for fast, free, deterministic tests.

Basic usage:
    import llmvcr

    @llmvcr.cassette("cassettes/my_test.yaml")
    def test_something():
        response = openai.chat.completions.create(...)
        assert "hello" in response.choices[0].message.content

Modes:
    "auto"     — record if cassette missing, replay if it exists (default)
    "record"   — always hit the real API and overwrite the cassette
    "playback" — always replay; raise if cassette is missing
"""

from .cassette import cassette
from .context import record, playback
from .errors import (
    LLMVCRError,
    CassetteNotFoundError,
    NoMatchFoundError,
    ProviderNotSupportedError,
)

__version__ = "0.1.1"
__all__ = [
    "cassette",
    "record",
    "playback",
    "LLMVCRError",
    "CassetteNotFoundError",
    "NoMatchFoundError",
    "ProviderNotSupportedError",
]