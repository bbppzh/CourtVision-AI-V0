"""Estimate a temporally confirmed proximity proxy, not physical possession."""

import math
from typing import TypedDict

from src.ball_tracker import BallTrack
from src.tracker import TrackedPerson


class PossessionProxy(TypedDict):
    """A visible track near the ball, with an uncalibrated proximity score."""

    player_track_id: int
    heuristic_score: float


def normalized_box_distance(center: tuple[float, float], bbox: list[int]) -> float:
    """Distance to the nearest rectangle point divided by the player box diagonal."""
    x, y = center
    x1, y1, x2, y2 = bbox
    dx, dy = max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy) / max(math.hypot(x2 - x1, y2 - y1), 1.0)


class PossessionEstimator:
    """Require consecutive proximity; give a confirmed holder a wider exit gate."""

    EXIT_THRESHOLD_MULTIPLIER = 1.5
    AMBIGUITY_MARGIN = 0.05

    def __init__(self, distance_threshold: float = 0.4, min_frames: int = 3) -> None:
        if not math.isfinite(distance_threshold) or not 0 < distance_threshold <= 1:
            raise ValueError("Possession distance threshold must be greater than 0 and at most 1.")
        if isinstance(min_frames, bool) or not isinstance(min_frames, int) or min_frames <= 0:
            raise ValueError("Possession minimum frames must be a positive integer.")
        self.threshold = distance_threshold
        self.min_frames = min_frames
        self._pending_id: int | None = None
        self._pending_count = 0
        self._holder_id: int | None = None
        self._last_frame: int | None = None
        self._ball_id: int | None = None

    def _clear(self) -> None:
        self._pending_id, self._holder_id = None, None
        self._pending_count = 0

    def update(
        self, ball: BallTrack | None, players: list[TrackedPerson], frame_index: int,
    ) -> PossessionProxy | None:
        """Return a confirmed visible proxy or unknown; never carry it across missing input."""
        if self._last_frame != frame_index - 1:
            self._clear()
        self._last_frame = frame_index
        if ball is None or not players:
            self._clear()
            self._ball_id = None
            return None
        if self._ball_id != ball["ball_track_id"]:
            self._clear()
        self._ball_id = ball["ball_track_id"]
        distances = sorted(
            (normalized_box_distance(ball["center"], person["bbox"]), person["track_id"])
            for person in players
        )
        best_distance, best_id = distances[0]
        # Overlapping/near-equal boxes cannot support confident attribution.
        ambiguous = len(distances) > 1 and distances[1][0] - best_distance < self.AMBIGUITY_MARGIN
        candidate_id = best_id if best_distance <= self.threshold and not ambiguous else None
        if candidate_id == self._pending_id and candidate_id is not None:
            self._pending_count += 1
        else:
            self._pending_id, self._pending_count = candidate_id, int(candidate_id is not None)
        if self._pending_count >= self.min_frames:
            self._holder_id = candidate_id
        if ambiguous:
            self._holder_id = None
        holder_distance = next((distance for distance, identity in distances if identity == self._holder_id), None)
        exit_threshold = self.threshold * self.EXIT_THRESHOLD_MULTIPLIER
        if holder_distance is None or holder_distance > exit_threshold:
            self._holder_id = None
            return None
        return {
            "player_track_id": self._holder_id,
            "heuristic_score": max(0.0, 1 - holder_distance / exit_threshold) * ball["confidence"],
        }
