"""
Handles reading and writing cassette files.

Cassette format (YAML):
    llmvcr_version: "0.1"
    provider: openai
    recorded_at: "2025-03-03T10:00:00Z"
    interactions:
      - request:
          model: gpt-4
          messages:
            - role: user
              content: Hello!
        response:
          id: chatcmpl-abc123
          choices:
            - message:
                role: assistant
                content: Hi there!
          usage:
            prompt_tokens: 10
            completion_tokens: 5
            total_tokens: 15
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

CASSETTE_VERSION = "0.1"


def load(path: str) -> Optional[Dict[str, Any]]:
    """Load a cassette from disk. Returns None if the file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save(path: str, provider: str, interactions: List[Dict[str, Any]]) -> None:
    """Save interactions to a cassette file, creating directories as needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    cassette = {
        "llmvcr_version": CASSETTE_VERSION,
        "provider": provider,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interactions": interactions,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cassette, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_interactions(cassette_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the interactions list from cassette data."""
    return cassette_data.get("interactions", [])