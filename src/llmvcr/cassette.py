"""
The @cassette decorator — the primary API for use in tests.

Usage:
    @llmvcr.cassette("cassettes/my_test.yaml")
    def test_something():
        response = openai.chat.completions.create(...)
        assert "hello" in response.choices[0].message.content

    # Explicit modes:
    @llmvcr.cassette("cassettes/my_test.yaml", mode="record")
    @llmvcr.cassette("cassettes/my_test.yaml", mode="playback")
    @llmvcr.cassette("cassettes/my_test.yaml", mode="auto")     # default
"""

import functools
from typing import Callable, Optional

from .recorder import Recorder
from ._patch import apply_patch


def cassette(
    path: str,
    mode: str = "auto",
    provider: str = "openai",
):
    """
    Decorator that wraps a function with llmvcr record/playback.

    Args:
        path:     Path to the cassette YAML file.
        mode:     "auto" (default), "record", or "playback".
                  - auto:     Record if cassette missing, replay if it exists.
                  - record:   Always call the real API and overwrite the cassette.
                  - playback: Always replay; raise CassetteNotFoundError if missing.
        provider: "openai" (default) or "anthropic".

    Returns:
        A decorator that can wrap any callable.

    Example:
        @llmvcr.cassette("cassettes/summarize.yaml")
        def test_summarize():
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Summarize AI"}],
            )
            assert "intelligence" in response.choices[0].message.content
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recorder = Recorder(path=path, mode=mode, provider=provider)
            record_cb, playback_cb = recorder.make_callbacks()
            with apply_patch(provider, record_cb, playback_cb):
                result = func(*args, **kwargs)
            recorder.save()
            return result
        return wrapper
    return decorator