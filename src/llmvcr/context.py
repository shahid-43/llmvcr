"""
Context manager API for llmvcr.

Usage:
    with llmvcr.record("cassettes/session.yaml"):
        response = openai.chat.completions.create(...)

    with llmvcr.playback("cassettes/session.yaml"):
        response = openai.chat.completions.create(...)
"""

from contextlib import contextmanager
from typing import Optional

from .recorder import Recorder
from ._patch import apply_patch


@contextmanager
def record(path: str, provider: str = "openai"):
    """
    Context manager: record a real API call and save to cassette.

    Args:
        path: Path to save the cassette YAML file.
        provider: "openai" (default) or "anthropic".

    Example:
        with llmvcr.record("cassettes/demo.yaml"):
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}]
            )
        # cassettes/demo.yaml is now written to disk
    """
    recorder = Recorder(path=path, mode="record", provider=provider)
    record_cb, playback_cb = recorder.make_callbacks()
    with apply_patch(provider, record_cb, playback_cb):
        yield recorder
    recorder.save()


@contextmanager
def playback(path: str, provider: str = "openai"):
    """
    Context manager: replay a cassette without hitting the real API.

    Args:
        path: Path to an existing cassette YAML file.
        provider: "openai" (default) or "anthropic".

    Raises:
        CassetteNotFoundError: if the cassette file does not exist.

    Example:
        with llmvcr.playback("cassettes/demo.yaml"):
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}]
            )
        # No API call was made
    """
    recorder = Recorder(path=path, mode="playback", provider=provider)
    record_cb, playback_cb = recorder.make_callbacks()
    with apply_patch(provider, record_cb, playback_cb):
        yield recorder