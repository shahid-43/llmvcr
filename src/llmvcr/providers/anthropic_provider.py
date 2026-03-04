"""
Anthropic provider support.

Intercepts calls to anthropic.messages.create() and handles record/playback.
Anthropic uses a slightly different message format than OpenAI, so we normalize
both into the same internal representation for matching purposes.
"""

from typing import Any, Dict
from unittest.mock import patch


# ── Serialization ──────────────────────────────────────────────────────────────

def serialize_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract meaningful fields from an anthropic.messages.create() call.
    Normalizes to the same internal format as the OpenAI provider.
    """
    messages = []

    # Anthropic supports a system parameter separately (not in messages list)
    if "system" in kwargs:
        messages.append({"role": "system", "content": kwargs["system"]})

    for m in kwargs.get("messages", []):
        content = m.get("content", "")
        # Anthropic content can be a string or a list of content blocks
        if isinstance(content, list):
            # Join text blocks for matching purposes
            content = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        messages.append({"role": m.get("role", ""), "content": content})

    return {
        "model": kwargs.get("model", ""),
        "messages": messages,
    }


def serialize_response(response: Any) -> Dict[str, Any]:
    """
    Convert an Anthropic Message object to a plain dict for YAML storage.
    Works with anthropic>=0.20 (Pydantic-based response objects).
    """
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "__dict__"):
        return vars(response)
    return {}


def deserialize_response(data: Dict[str, Any]) -> Any:
    """
    Reconstruct an Anthropic Message-like object from stored data.
    Caller can access .content[0].text as they would with a real response.
    """
    try:
        from anthropic.types import Message
        return Message.model_validate(data)
    except ImportError:
        raise ImportError(
            "anthropic package is required for Anthropic cassette playback.\n"
            "Install it with: pip install llmvcr[anthropic]"
        )
    except Exception:
        return _dict_to_namespace(data)


# ── Patching ───────────────────────────────────────────────────────────────────

def make_patch(record_callback, playback_callback):
    """
    Return a context manager that patches anthropic.Anthropic().messages.create.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not found. Install it with: pip install llmvcr[anthropic]"
        )

    original_create = anthropic.Anthropic.messages.create

    def patched_create(self_or_client, *args, **kwargs):
        request = serialize_request(kwargs)

        if playback_callback is not None:
            response_data = playback_callback(request)
            return deserialize_response(response_data)
        else:
            response = original_create(self_or_client, *args, **kwargs)
            response_data = serialize_response(response)
            record_callback(request, response_data)
            return response

    return patch.object(anthropic.resources.Messages, "create", patched_create)


# ── Helpers ────────────────────────────────────────────────────────────────────

class _Namespace:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _dict_to_namespace(v) if isinstance(v, dict) else
                    [_dict_to_namespace(i) if isinstance(i, dict) else i for i in v]
                    if isinstance(v, list) else v)


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return _Namespace(d)
    return d