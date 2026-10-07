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

    The association gate is a time-based speed limit, not a per-frame distance:
    a ball moving at a fixed speed covers more pixels per frame as the source
    frame rate drops, so a pixels-per-frame limit would reject the same physical
    flight at 24 FPS while accepting it at 120 FPS.
    """

    # Nominal 24 FPS ceiling used when a caller does not report a frame rate.
    DEFAULT_FPS = 24.0
    # Speed limit for one prediction step, as a fraction of the frame diagonal
    # per second. At the nominal 24 FPS this is the previous 12% of the diagonal
    # per frame, so single-frame association behaviour is unchanged.
    MOTION_GATE_FRACTION = 0.12
    # The gate also grows with the observed ball speed, so a fast flight is not
    # rejected when it slows down or changes direction between frames.
    SPEED_MARGIN_FRACTION = 0.75
    # How far ahead the constant-velocity prediction is trusted, in seconds.
    MAX_PREDICTION_SECONDS = 0.15
    MAX_AREA_FRACTION = 0.05
    MAX_ASPECT_RATIO = 3.0

    def __init__(
        self, width: int, height: int, history_length: int = 40, max_gap_frames: int = 5,
        fps: float = DEFAULT_FPS,
    ) -> None:
        if width <= 0 or height <= 0 or history_length <= 0 or max_gap_frames <= 0:
            raise ValueError("Ball tracker dimensions and history/gap lengths must be positive.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Ball tracker FPS must be finite and positive.")
        self.width = width
        self.height = height
        self.fps = fps
        self.motion_gate = math.hypot(width, height) * self.MOTION_GATE_FRACTION
        # Pixels per second a ball may plausibly travel with no observed motion.
        self.max_speed = self.motion_gate * self.DEFAULT_FPS
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

    def _prediction_gate(self, gap_frames: int) -> float:
        """How far the ball may travel during a gap, in pixels.

        The allowance is a speed limit times the elapsed seconds, so a fixed
        physical speed fits a long gap at a low frame rate and a short gap at a
        high one. The speed limit itself rises with the last observed speed,
        because a single position cannot predict a change of direction. The
        horizon cap keeps a long gap from justifying an arbitrarily distant jump.
        """
        seconds = min(gap_frames / self.fps, self.MAX_PREDICTION_SECONDS)
        observed = math.hypot(*self._velocity) * self.fps
        speed_limit = max(self.max_speed, observed * (1 + self.SPEED_MARGIN_FRACTION))
        return speed_limit * seconds

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
            # A missing observation ends the trail: history only ever contains
            # observed centers, so a gap is never drawn as if it were seen.
            self.history.clear()
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
            gate = self._prediction_gate(gap)
            gated = [candidate for candidate in candidates if distance(candidate) <= gate]
            if not gated:
                return None
            chosen = min(gated, key=lambda item: (distance(item), -item["confidence"], tuple(item["bbox"])))
            track_id = self._last["ball_track_id"]
            if gap == 1:
                self._velocity = tuple(chosen["center"][i] - self._last["center"][i] for i in (0, 1))
            else:
                # A gap ends the frame-to-frame estimate; the next observation
                # starts a fresh one instead of extrapolating old motion.
                self._velocity = (0.0, 0.0)
                self.history.clear()
        track: BallTrack = {**chosen, "ball_track_id": track_id}
        self._last, self._last_frame = track, frame_index
        self.history.append((frame_index, *track["center"]))
        return track
