"""
Ollama provider support — runs Llama (and other models) locally.

Intercepts calls to the ollama Python client:
    import ollama
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Ollama's response format uses response['message']['content'] (dict-based),
which is slightly different from OpenAI's object-based response.

Use this provider when running Llama, Mistral, Qwen, or any other model
locally via Ollama (https://ollama.com).

Install: pip install llmvcr[ollama]
"""

from typing import Any, Dict
from unittest.mock import patch


# ── Serialization ──────────────────────────────────────────────────────────────

def serialize_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract meaningful fields from an ollama.chat() call.
    Normalizes to llmvcr's internal format.
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
    Serialize an Ollama response to a plain dict.

    Ollama returns either a dict or a ChatResponse object (ollama>=0.2).
    We normalize both to a consistent dict shape.
    """
    # ollama>=0.2 returns a ChatResponse object
    if hasattr(response, "message"):
        msg = response.message
        return {
            "model": getattr(response, "model", ""),
            "message": {
                "role": getattr(msg, "role", "assistant"),
                "content": getattr(msg, "content", ""),
            },
            "done": getattr(response, "done", True),
            "prompt_eval_count": getattr(response, "prompt_eval_count", 0),
            "eval_count": getattr(response, "eval_count", 0),
        }

    # dict-style response (older ollama or direct REST)
    if isinstance(response, dict):
        return response

    return vars(response) if hasattr(response, "__dict__") else {}


def deserialize_response(data: Dict[str, Any]) -> Any:
    """
    Reconstruct an Ollama-like response from stored data.
    Callers can use response['message']['content'] or response.message.content.
    """
    try:
        from ollama import ChatResponse, Message
        msg_data = data.get("message", {})
        msg = Message(role=msg_data.get("role", "assistant"), content=msg_data.get("content", ""))
        return ChatResponse(
            model=data.get("model", ""),
            message=msg,
            done=data.get("done", True),
        )
    except (ImportError, Exception):
        # Fallback: return a simple namespace that supports both
        # response['message']['content'] and response.message.content
        return _OllamaResponseProxy(data)


# ── Patching ───────────────────────────────────────────────────────────────────

def make_patch(record_callback, playback_callback):
    """
    Return a context manager that patches ollama.chat.
    """
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "ollama package not found.\n"
            "Install it with: pip install llmvcr[ollama]\n"
            "Also make sure Ollama is running locally: https://ollama.com"
        )

    original_chat = ollama.chat

    def patched_chat(*args, **kwargs):
        # ollama.chat can be called positionally: ollama.chat("model", messages=[...])
        if args:
            kwargs["model"] = args[0]

        request = serialize_request(kwargs)

        if playback_callback is not None:
            response_data = playback_callback(request)
            return deserialize_response(response_data)
        else:
            response = original_chat(*args, **kwargs)
            response_data = serialize_response(response)
            record_callback(request, response_data)
            return response

    return patch.object(ollama, "chat", patched_chat)


# ── Helpers ────────────────────────────────────────────────────────────────────

class _OllamaResponseProxy:
    """
    Proxy object that lets callers use EITHER dict-style OR attribute-style access.

    response["message"]["content"]   ← dict style
    response.message.content         ← attribute style
    Both work.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        msg_data = data.get("message", {})
        self.message = _MessageProxy(msg_data)
        self.model = data.get("model", "")
        self.done = data.get("done", True)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


class _MessageProxy:
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self.role = data.get("role", "assistant")
        self.content = data.get("content", "")

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)