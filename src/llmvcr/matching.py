"""
Semantic request matching.

Instead of matching on the raw HTTP body (like VCR.py does), llmvcr matches
on the meaningful parts of an LLM request:
  - model name
  - message content and roles

Parameters like temperature, max_tokens, top_p etc. are intentionally ignored
by default — they don't change WHICH response you'd want to replay.

This means adding `temperature=0.7` to an existing call won't break your cassette.
"""

from typing import Any, Dict, List, Optional


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Strip extra fields, keep only role and content."""
    normalized = []
    for msg in messages:
        normalized.append({
            "role": str(msg.get("role", "")).strip().lower(),
            "content": str(msg.get("content", "")).strip(),
        })
    return normalized


def _request_key(request: Dict[str, Any]) -> tuple:
    """
    Build a hashable key from the semantically meaningful parts of a request.
    Two requests with the same model + messages will match, regardless of
    temperature, max_tokens, etc.
    """
    model = str(request.get("model", "")).strip().lower()
    messages = _normalize_messages(request.get("messages", []))
    # Convert to a hashable structure
    messages_key = tuple(
        (m["role"], m["content"]) for m in messages
    )
    return (model, messages_key)


def find_match(
    request: Dict[str, Any],
    interactions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Find the first interaction in the cassette whose request matches
    the incoming request semantically.

    Returns the matching interaction dict, or None if no match found.
    """
    incoming_key = _request_key(request)

    for interaction in interactions:
        cassette_request = interaction.get("request", {})
        cassette_key = _request_key(cassette_request)
        if incoming_key == cassette_key:
            return interaction

    return None


def request_summary(request: Dict[str, Any]) -> str:
    """Human-readable summary of a request for error messages."""
    model = request.get("model", "unknown")
    messages = request.get("messages", [])
    last_msg = messages[-1] if messages else {}
    role = last_msg.get("role", "?")
    content = last_msg.get("content", "")
    if len(content) > 80:
        content = content[:77] + "..."
    return f"model={model}, last_message=[{role}]: {content!r}"