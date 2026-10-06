"""A small continuity-based matcher for one primary sports-ball candidate."""

import math
from collections import deque

from src.detector import BallDetection


class BallTrack(BallDetection):
    """A visible ball observation with an ID separate from player track IDs."""

    ball_track_id: int


class BallTracker:
    """Choose one plausible ball, using motion continuity before confidence.

    No positions are fabricated during occlusion. Short gaps retain association
    state, but drawing history restarts across any missing observation.
    """

    MOTION_GATE_FRACTION = 0.12
    MAX_AREA_FRACTION = 0.05
    MAX_ASPECT_RATIO = 3.0

    def __init__(self, width: int, height: int, history_length: int = 40, max_gap_frames: int = 5) -> None:
        if width <= 0 or height <= 0 or history_length <= 0 or max_gap_frames <= 0:
            raise ValueError("Ball tracker dimensions and history/gap lengths must be positive.")
        self.width = width
        self.height = height
        self.motion_gate = math.hypot(width, height) * self.MOTION_GATE_FRACTION
        self.max_gap_frames = max_gap_frames
        self.history: deque[tuple[int, float, float]] = deque(maxlen=history_length)
        self.reset()

    def reset(self) -> None:
        """Reset the logical ball IDs and all association state for a new video."""
        self.history.clear()
        self._last: BallTrack | None = None
        self._last_frame: int | None = None
        self._velocity = (0.0, 0.0)
        self._next_id = 1

    def _plausible(self, candidate: BallDetection) -> bool:
        """Reject extreme scene geometry; this does not prove basketball identity."""
        x1, y1, x2, y2 = candidate["bbox"]
        width, height = x2 - x1, y2 - y1
        x, y = candidate["center"]
        return (
            width > 0 and height > 0 and 1 / self.MAX_ASPECT_RATIO <= width / height <= self.MAX_ASPECT_RATIO
            and width * height <= self.width * self.height * self.MAX_AREA_FRACTION
            and math.isfinite(x) and math.isfinite(y)
            and math.isfinite(candidate["confidence"])
            and 0 <= x < self.width and 0 <= y < self.height
        )

    def update(self, detections: list[BallDetection], frame_index: int) -> BallTrack | None:
        """Return one visible candidate or unknown; avoid instant switches to distant balls."""
        if self._last_frame is not None and frame_index - self._last_frame > self.max_gap_frames:
            self._last = None
            self._velocity = (0.0, 0.0)
            self.history.clear()
        candidates = [candidate for candidate in detections if self._plausible(candidate)]
        if not candidates:
            return None
        if self._last is None:
            chosen = min(candidates, key=lambda item: (-item["confidence"], tuple(item["bbox"])))
            track_id = self._next_id
            self._next_id += 1
        else:
            gap = frame_index - self._last_frame
            predicted = tuple(self._last["center"][i] + self._velocity[i] * gap for i in (0, 1))
            def distance(candidate: BallDetection) -> float:
                return math.dist(candidate["center"], predicted)
            gated = [candidate for candidate in candidates if distance(candidate) <= self.motion_gate * min(gap, 3)]
            if not gated:
                return None
            chosen = min(gated, key=lambda item: (distance(item), -item["confidence"], tuple(item["bbox"])))
            track_id = self._last["ball_track_id"]
            if gap == 1:
                self._velocity = tuple(chosen["center"][i] - self._last["center"][i] for i in (0, 1))
            else:
                self._velocity = (0.0, 0.0)
                self.history.clear()
        track: BallTrack = {**chosen, "ball_track_id": track_id}
        self._last, self._last_frame = track, frame_index
        self.history.append((frame_index, *track["center"]))
        return track
