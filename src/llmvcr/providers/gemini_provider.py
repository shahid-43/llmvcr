"""
Google Gemini provider support.

Intercepts calls to the google-genai SDK:
    client.models.generate_content(model=..., contents=..., config=...)

The Gemini SDK uses a different message format from OpenAI:
  - Messages are called "contents" (not "messages")
  - Content can be a plain string or a list of Part objects
  - System instructions are passed via GenerateContentConfig, not in messages

We normalize everything into the same internal format used by all llmvcr
providers so cassettes stay consistent and matching works across providers.

Install: pip install llmvcr[gemini]
"""

from typing import Any, Dict, List
from unittest.mock import patch


# ── Serialization ──────────────────────────────────────────────────────────────

def _normalize_contents(contents: Any) -> List[Dict[str, str]]:
    """
    Normalize Gemini's 'contents' field into the internal messages format.

    Gemini accepts:
      - A plain string: "Hello!"
      - A list of strings or Part objects
      - A list of Content objects with role + parts
    """
    messages = []

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return messages

    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                parts = item.get("parts", [])
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in parts
                ) if parts else item.get("text", "")
                messages.append({"role": role, "content": content})
            elif hasattr(item, "role") and hasattr(item, "parts"):
                # Content object from SDK
                parts_text = " ".join(
                    p.text for p in item.parts
                    if hasattr(p, "text") and p.text
                )
                messages.append({"role": item.role, "content": parts_text})
            elif hasattr(item, "text"):
                messages.append({"role": "user", "content": item.text})

    return messages


def serialize_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract meaningful fields from a client.models.generate_content() call.
    Normalizes to llmvcr's internal format for consistent matching.
    """
    model = kwargs.get("model", "")
    contents = kwargs.get("contents", "")
    config = kwargs.get("config", None)

    messages = _normalize_contents(contents)

    # Extract system instruction from config if present
    if config is not None:
        system = None
        if isinstance(config, dict):
            system = config.get("system_instruction")
        elif hasattr(config, "system_instruction"):
            system = config.system_instruction

        if system:
            system_text = system if isinstance(system, str) else str(system)
            messages.insert(0, {"role": "system", "content": system_text})

    return {"model": model, "messages": messages}


def serialize_response(response: Any) -> Dict[str, Any]:
    """
    Convert a Gemini GenerateContentResponse to a plain dict for YAML storage.
    """
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "__dict__"):
        return vars(response)

    # Build a minimal dict manually from the response object
    result = {}
    if hasattr(response, "text"):
        result["text"] = response.text
    if hasattr(response, "candidates"):
        result["candidates"] = [
            {
                "content": {
                    "parts": [{"text": p.text} for p in c.content.parts if hasattr(p, "text")],
                    "role": c.content.role if hasattr(c.content, "role") else "model",
                },
                "finish_reason": str(c.finish_reason) if hasattr(c, "finish_reason") else "STOP",
            }
            for c in response.candidates
        ]
    if hasattr(response, "usage_metadata"):
        um = response.usage_metadata
        result["usage_metadata"] = {
            "prompt_token_count": getattr(um, "prompt_token_count", 0),
            "candidates_token_count": getattr(um, "candidates_token_count", 0),
            "total_token_count": getattr(um, "total_token_count", 0),
        }
    return result


def deserialize_response(data: Dict[str, Any]) -> Any:
    """
    Reconstruct a Gemini-like response object from stored data.
    Callers can access .text or .candidates[0].content.parts[0].text
    as they would with a real response.
    """
    try:
        from google.genai.types import GenerateContentResponse
        return GenerateContentResponse.model_validate(data)
    except Exception:
        return _dict_to_namespace(data)


# ── Patching ───────────────────────────────────────────────────────────────────

def make_patch(record_callback, playback_callback):
    """
    Return a context manager that patches google.genai.Client.models.generate_content.
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package not found.\n"
            "Install it with: pip install llmvcr[gemini]"
        )

    original_generate = genai.Client.models.generate_content

    def patched_generate(self_or_models, *args, **kwargs):
        request = serialize_request(kwargs)

        if playback_callback is not None:
            response_data = playback_callback(request)
            return deserialize_response(response_data)
        else:
            response = original_generate(self_or_models, *args, **kwargs)
            response_data = serialize_response(response)
            record_callback(request, response_data)
            return response

    return patch.object(genai.Client.models.__class__, "generate_content", patched_generate)


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

    @property
    def text(self):
        """Convenience: return text from the first candidate's first part."""
        candidates = getattr(self, "candidates", [])
        if candidates:
            c = candidates[0]
            parts = getattr(getattr(c, "content", None), "parts", [])
            if parts:
                return getattr(parts[0], "text", "")
        return getattr(self, "_text", "")


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return _Namespace(d)
    return d