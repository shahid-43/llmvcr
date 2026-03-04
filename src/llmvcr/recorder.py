"""
The Recorder is the central engine of llmvcr.

It holds the cassette state (loaded interactions, newly recorded interactions),
handles matching incoming requests to stored ones, and decides whether to
call the real API or return a stored response.

One Recorder instance is created per cassette context (per @cassette decorator
or per `with record()/playback()` block).
"""

from typing import Any, Dict, List, Optional

from . import storage, matching
from .errors import CassetteNotFoundError, NoMatchFoundError


class Recorder:
    """
    Manages a single cassette session.

    Args:
        path: Path to the cassette YAML file.
        mode: One of "auto", "record", "playback".
        provider: "openai" or "anthropic".
    """

    def __init__(self, path: str, mode: str, provider: str):
        self.path = path
        self.mode = mode
        self.provider = provider

        # Interactions loaded from disk (for playback)
        self._stored: List[Dict[str, Any]] = []
        # Interactions captured this session (for recording)
        self._recorded: List[Dict[str, Any]] = []
        # Whether we need to write to disk when done
        self._dirty = False

        self._load()

    def _load(self):
        """Load existing cassette from disk if it exists."""
        data = storage.load(self.path)
        if data is not None:
            self._stored = storage.get_interactions(data)
        elif self.mode == "playback":
            raise CassetteNotFoundError(self.path)

    def should_record(self) -> bool:
        """Return True if this session should make real API calls."""
        if self.mode == "record":
            return True
        if self.mode == "playback":
            return False
        # auto mode: record if no cassette on disk, playback if it exists
        return len(self._stored) == 0

    def on_response(self, request: Dict[str, Any], response: Dict[str, Any]):
        """Called after a real API call. Stores the interaction."""
        self._recorded.append({"request": request, "response": response})
        self._dirty = True

    def get_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find a stored response for the given request.
        Raises NoMatchFoundError if none is found.
        """
        # Search in already-recorded interactions first (same session, multi-call)
        all_interactions = self._stored + self._recorded
        interaction = matching.find_match(request, all_interactions)

        if interaction is None:
            raise NoMatchFoundError(
                self.path,
                matching.request_summary(request)
            )

        return interaction["response"]

    def save(self):
        """Write all recorded interactions to disk if anything was captured."""
        if not self._dirty:
            return

        # In record mode, replace stored. In auto mode, append new ones.
        if self.mode == "record":
            interactions = self._recorded
        else:
            interactions = self._stored + self._recorded

        storage.save(self.path, self.provider, interactions)
        self._dirty = False

    def make_callbacks(self):
        """
        Return (record_callback, playback_callback) to pass to a provider patch.

        - record_callback: called with (request, response) after a real API call
        - playback_callback: called with (request,) instead of real API call
          Returns the stored response dict.
          Is None when we're in record mode (don't intercept).
        """
        if self.should_record():
            return self.on_response, None
        else:
            return None, self.get_response