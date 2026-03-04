"""
Internal routing layer: picks the right provider patch to apply.
"""

from contextlib import contextmanager
from typing import Callable, Optional

from .errors import ProviderNotSupportedError


@contextmanager
def apply_patch(provider: str, record_callback, playback_callback):
    """
    Apply the appropriate provider patch as a context manager.

    Args:
        provider: "openai" or "anthropic"
        record_callback: called with (request, response) after a real API call
        playback_callback: called with (request,) -> response, or None in record mode
    """
    if provider == "openai":
        from .providers.openai_provider import make_patch
    elif provider == "anthropic":
        from .providers.anthropic_provider import make_patch
    else:
        raise ProviderNotSupportedError(provider)

    patch_ctx = make_patch(record_callback, playback_callback)
    with patch_ctx:
        yield