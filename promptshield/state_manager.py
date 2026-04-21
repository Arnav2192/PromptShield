"""
Session state manager: tracks cumulative risk scores across conversation turns.
Uses exponential decay to weight recent turns more heavily.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field


class StateManager:
    """
    Thread-safe cumulative risk score tracker.

    Score formula per turn: new_score = previous_score * decay + current_score

    Attributes
    ----------
    decay : float
        Decay factor applied to the previous score each turn (0.0–1.0).
        Lower values forget history faster.
    threshold : float
        Cumulative score above which a session is considered flagged.
    """

    def __init__(self, decay: float = 0.6, threshold: float = 1.5) -> None:
        self.decay = decay
        self.threshold = threshold
        self._scores: dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, session_id: str, current_score: float) -> float:
        """Apply decay formula and return the new cumulative score."""
        with self._lock:
            prev = self._scores.get(session_id, 0.0)
            new_score = prev * self.decay + current_score
            self._scores[session_id] = new_score
            return new_score

    def get_score(self, session_id: str) -> float:
        """Return current cumulative score for a session (0.0 if unknown)."""
        with self._lock:
            return self._scores.get(session_id, 0.0)

    def reset(self, session_id: str) -> None:
        """Reset a session's cumulative score to 0.0."""
        with self._lock:
            self._scores[session_id] = 0.0

    def is_flagged(self, session_id: str) -> bool:
        """Return True if the session's cumulative score meets or exceeds threshold."""
        return self.get_score(session_id) >= self.threshold
