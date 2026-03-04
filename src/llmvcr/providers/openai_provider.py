"""
OpenAI provider support.

Intercepts calls to openai.chat.completions.create() and:
  - In record mode: calls the real API, serializes the response, stores it
  - In playback mode: deserializes a stored response and returns it directly

We patch at the SDK method level, never touching the HTTP layer.
This means: no gzip issues, no auth headers, no streaming problems.
"""

from typing import Any, Dict
from unittest.mock import patch


# ── Serialization ──────────────────────────────────────────────────────────────

def serialize_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the semantically meaningful fields from an openai.create() call.
    Intentionally drops: temperature, max_tokens, top_p, stream, etc.
    These don't affect which response we'd want to replay.
    """
    return {
        "model": kwargs.get("model", ""),
        "messages": [
            {"role": m.get("role", ""), "content": m.get("content", "")}
            for m in kwargs.get("messages", [])
        ],
    }


def serialize_response(response: Any) -> Dict[str, Any]:
    """
    Convert an OpenAI ChatCompletion object into a plain dict for YAML storage.
    Works with openai>=1.0 (Pydantic-based response objects).
    """
    # openai>=1.0 objects have .model_dump() from Pydantic
    if hasattr(response, "model_dump"):
        return response.model_dump()
    # Fallback: try __dict__
    if hasattr(response, "__dict__"):
        return vars(response)
    return {}


def deserialize_response(data: Dict[str, Any]) -> Any:
    """
    Reconstruct an OpenAI ChatCompletion-like object from a stored dict.
    We use openai's own types so the caller can access .choices[0].message.content
    exactly as they would with a real response.
    """
    try:
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(data)
    except ImportError:
        raise ImportError(
            "openai package is required for OpenAI cassette playback.\n"
            "Install it with: pip install llmvcr[openai]"
        )
    except Exception:
        # Fallback: return a simple namespace object
        return _dict_to_namespace(data)


# ── Patching ───────────────────────────────────────────────────────────────────

def make_patch(record_callback, playback_callback):
    """
    Return a context manager that patches openai.chat.completions.create.

    record_callback(request_dict, response_dict) — called after a real API call
    playback_callback(request_dict) -> response_dict — called instead of real API
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package not found. Install it with: pip install llmvcr[openai]"
        )

    original_create = openai.chat.completions.create

    def patched_create(*args, **kwargs):
        request = serialize_request(kwargs)

        if playback_callback is not None:
            # Playback mode: return stored response
            response_data = playback_callback(request)
            return deserialize_response(response_data)
        else:
            # Record mode: call real API, capture and store response
            response = original_create(*args, **kwargs)
            response_data = serialize_response(response)
            record_callback(request, response_data)
            return response

    return patch.object(openai.chat.completions, "create", patched_create)


# ── Helpers ────────────────────────────────────────────────────────────────────

class _Namespace:
    """Simple recursive namespace for dict-to-object fallback."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _dict_to_namespace(v) if isinstance(v, dict) else
                    [_dict_to_namespace(i) if isinstance(i, dict) else i for i in v]
                    if isinstance(v, list) else v)


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return _Namespace(d)
    return d