"""Keep short player trails without joining points across missing frames."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from src.tracker import TrackedPerson


class TrackHistory:
    """Bound each trail and remove IDs after too many unseen frames.

    Memory scales with recent track IDs times ``max_length``. Histories expire
    when more than ``max_stale_frames`` frames have passed since an observation.
    """

    def __init__(self, max_length: int = 40, max_stale_frames: int = 60) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if (
            isinstance(max_stale_frames, bool)
            or not isinstance(max_stale_frames, int)
            or max_stale_frames <= 0
        ):
            raise ValueError("max_stale_frames must be a positive integer.")

        self.max_length = max_length
        self.max_stale_frames = max_stale_frames
        self.histories: dict[int, deque[tuple[float, float]]] = {}
        self._last_seen: dict[int, int] = {}

    def update(self, tracks: Iterable[TrackedPerson], frame_index: int) -> None:
        """Add this frame's centers, starting a fresh trail after any gap."""
        for track in tracks:
            track_id = track["track_id"]
            if track_id not in self.histories:
                self.histories[track_id] = deque(maxlen=self.max_length)
            elif self._last_seen[track_id] != frame_index - 1:
                self.histories[track_id].clear()

            self.histories[track_id].append(track["center"])
            self._last_seen[track_id] = frame_index

        self._cleanup(frame_index)

    def get_points(self, track_id: int) -> list[tuple[float, float]]:
        """Return a copy of a trail, or an empty list for an unknown ID."""
        return list(self.histories.get(track_id, ()))

    def _cleanup(self, frame_index: int) -> None:
        """Run on every frame, including frames with no visible players."""
        stale_ids = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if frame_index - last_seen > self.max_stale_frames
        ]
        for track_id in stale_ids:
            del self.histories[track_id]
            del self._last_seen[track_id]
