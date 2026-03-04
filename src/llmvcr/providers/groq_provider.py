"""
Groq provider support — runs Llama (and other models) via Groq's cloud API.

Intercepts calls to the Groq Python client:
    from groq import Groq
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Groq's SDK is intentionally OpenAI-compatible — same method names, same
response shape, same message format. This means the serialization logic
is nearly identical to the OpenAI provider, and cassettes from both are
interchangeable at the data level.

Supported Llama models on Groq (as of 2025):
    llama-3.3-70b-versatile
    llama-3.1-8b-instant
    llama-3.1-70b-versatile
    llama3-8b-8192
    llama3-70b-8192

Install: pip install llmvcr[groq]
"""

from typing import Any, Dict
from unittest.mock import patch


# ── Serialization ──────────────────────────────────────────────────────────────

def serialize_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract meaningful fields from a groq client.chat.completions.create() call.
    Identical structure to OpenAI — model + messages, drop sampling params.
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
    Convert a Groq ChatCompletion response to a plain dict.
    Groq uses Pydantic models identical in shape to OpenAI's.
    """
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "__dict__"):
        return vars(response)
    return {}


def deserialize_response(data: Dict[str, Any]) -> Any:
    """
    Reconstruct a Groq ChatCompletion-like response from stored data.
    Callers access .choices[0].message.content as with a real response.
    """
    try:
        from groq.types.chat import ChatCompletion
        return ChatCompletion.model_validate(data)
    except ImportError:
        raise ImportError(
            "groq package not found.\n"
            "Install it with: pip install llmvcr[groq]"
        )
    except Exception:
        return _dict_to_namespace(data)


# ── Patching ───────────────────────────────────────────────────────────────────

def make_patch(record_callback, playback_callback):
    """
    Return a context manager that patches groq.resources.chat.completions.Completions.create.
    """
    try:
        import groq
    except ImportError:
        raise ImportError(
            "groq package not found.\n"
            "Install it with: pip install llmvcr[groq]"
        )

    original_create = groq.resources.chat.completions.Completions.create

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

    return patch.object(
        groq.resources.chat.completions.Completions,
        "create",
        patched_create,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

class _Namespace:
    def __init__(self, d):
        for k, v in d.items():
            setattr(
                self, k,
                _dict_to_namespace(v) if isinstance(v, dict) else
                [_dict_to_namespace(i) if isinstance(i, dict) else i for i in v]
                if isinstance(v, list) else v
            )


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return _Namespace(d)
    return d