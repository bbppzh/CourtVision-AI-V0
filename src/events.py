"""Conservative temporal shot candidates around a manually selected rim ROI."""

import math
from collections import deque
from dataclasses import dataclass
from typing import TypedDict

from src.ball_tracker import BallTrack
from src.interaction import PossessionProxy, normalized_box_distance
from src.tracker import TrackedPerson


@dataclass(frozen=True)
class RimROI:
    """An image rectangle selected manually; no hoop model or calibration."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("Rim ROI needs four integers: x1,y1,x2,y2.")
        if self.x1 < 0 or self.y1 < 0 or self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Rim ROI needs nonnegative coordinates with x2 > x1 and y2 > y1.")

    @classmethod
    def parse(cls, value: str) -> "RimROI":
        """Parse the CLI's comma-separated image coordinates."""
        try:
            coordinates = [int(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError("Rim ROI needs four integers: x1,y1,x2,y2.") from exc
        if len(coordinates) != 4:
            raise ValueError("Rim ROI needs four integers: x1,y1,x2,y2.")
        return cls(*coordinates)

    def validate_dimensions(self, width: int, height: int) -> None:
        if self.x2 > width or self.y2 > height:
            raise ValueError(f"Rim ROI must lie within the {width}x{height} video.")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def distance(self, center: tuple[float, float]) -> float:
        x, y = center
        return math.hypot(max(self.x1 - x, 0, x - self.x2), max(self.y1 - y, 0, y - self.y2))


class Event(TypedDict):
    """One zero-based frame decision in the exported candidate timeline."""

    event_id: int
    type: str
    frame: int
    timestamp_seconds: float
    player_track_id: int | None
    ball_track_id: int | None
    shot_id: int | None
    heuristic_score: float


class ShotCandidate(TypedDict):
    """A temporal attempt and its conservative, possibly unknown outcome."""

    shot_id: int
    start_frame: int
    decision_frame: int
    player_track_id: int | None
    ball_track_id: int
    outcome: str
    outcome_frame: int | None
    heuristic_score: float
    outcome_heuristic_score: float


@dataclass
class _Holder:
    frame: int
    player_id: int
    ball_id: int
    bbox: list[int]


@dataclass
class _Release:
    frame: int
    ball_id: int
    center: tuple[float, float]
    player_id: int | None
    rim_distance: float


@dataclass
class _Shot:
    report: ShotCandidate
    last_seen: int
    above_seen: bool = False
    rim_crossed: bool = False
    below_frames: int = 0
    outside_frames: int = 0
    # Where this ball currently sits relative to the rim ROI, for the pass state
    # machine: "unknown", "above", "inside", or "below". Only a continuous
    # downward walk from "above" through "inside" to "below" confirms a crossing.
    pass_state: str = "unknown"


class EventAnalyzer:
    """POSSESSED → RELEASED → ASCENDING → SHOT_CANDIDATE → outcome.

    Scores are fixed rule-strength labels, not calibrated probabilities.
    Missing observations cannot establish a rim crossing or a made candidate.

    Vertical direction is measured in image pixels per second, so one speed
    threshold means the same thing at every source frame rate. Positions are
    never interpolated across a missing observation; a gap only ends the
    current motion and crossing evidence.
    """

    ATTEMPT_SCORE = 0.75
    MADE_SCORE = 0.85
    MISSED_SCORE = 0.65
    SHOT_TIMEOUT_SECONDS = 2.0
    COOLDOWN_SECONDS = 0.75
    MISSING_GRACE_SECONDS = 0.12
    OVERLAY_SECONDS = 0.8
    RELEASE_WINDOW_SECONDS = 0.75
    ATTRIBUTION_WINDOW_SECONDS = 0.4
    # Vertical speed needed to call a direction, in image pixels per second.
    # A fixed pixels-per-frame epsilon would mean a different physical speed at
    # every frame rate, so the threshold is time based instead.
    DIRECTION_MIN_SPEED_PX_PER_SECOND = 36.0
    # Rolling window whose least-squares slope estimates vertical speed.
    # Averaging single-frame pixel differences is noisy at high frame rates;
    # a slope fitted over a window is stable from 24 to 120 FPS.
    DIRECTION_SMOOTHING_SECONDS = 0.08
    DIRECTION_MIN_SAMPLES = 3
    # A direction needs several observations spread over a real time span, not
    # a fixed frame count, so jitter cannot confirm itself faster at high FPS.
    DIRECTION_CONFIRM_SECONDS = 0.04
    MIN_OUTCOME_FRAMES = 2

    def __init__(self, fps: float, rim_roi: RimROI | None = None, possession_threshold: float = 0.4) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Event FPS must be finite and positive.")
        self.fps, self.rim_roi = fps, rim_roi
        self.possession_threshold = possession_threshold
        self.timeout_frames = max(1, math.ceil(fps * self.SHOT_TIMEOUT_SECONDS))
        self.cooldown_frames = max(1, math.ceil(fps * self.COOLDOWN_SECONDS))
        self.max_missing_frames = max(1, math.ceil(fps * self.MISSING_GRACE_SECONDS))
        self.overlay_frames = max(1, math.ceil(fps * self.OVERLAY_SECONDS))
        # Keep enough samples for a median over the smoothing duration.
        self.motion_window = max(
            self.DIRECTION_MIN_SAMPLES, round(fps * self.DIRECTION_SMOOTHING_SECONDS),
        )
        self.events: list[Event] = []
        self.shots: list[ShotCandidate] = []
        self.state = "IDLE"
        self._holder: _Holder | None = None
        self._release: _Release | None = None
        self._active: _Shot | None = None
        self._previous_ball: BallTrack | None = None
        self._previous_frame: int | None = None
        # Consecutive (frame, y) observations used for the time-based slope.
        self._observations: deque[tuple[int, float]] = deque(maxlen=self.motion_window)
        self._proxy_id: int | None = None
        self._cooldown_until = -1
        self._last_overlay: tuple[int, str] | None = None

    def _observe_motion(self, ball: BallTrack, frame_index: int) -> None:
        """Record one visible center; the caller clears this across gaps."""
        self._observations.append((frame_index, ball["center"][1]))

    def _vertical_speed(self) -> float | None:
        """Vertical velocity in pixels per second, or unknown.

        The estimate is a least-squares slope over the retained observations, so
        it does not depend on the sampling interval: the same physical speed
        gives the same value at 24 and at 120 FPS. Gaps clear the window instead
        of being bridged, and a window that is too short or too brief stays
        undecided rather than reporting a confident direction.
        """
        if len(self._observations) < max(2, self.DIRECTION_MIN_SAMPLES):
            return None
        mean_frame = sum(frame for frame, _ in self._observations) / len(self._observations)
        mean_y = sum(y for _, y in self._observations) / len(self._observations)
        variance = sum((frame - mean_frame) ** 2 for frame, _ in self._observations)
        if variance <= 0:
            return None
        covariance = sum(
            (frame - mean_frame) * (y - mean_y) for frame, y in self._observations
        )
        elapsed = self._observations[-1][0] - self._observations[0][0]
        if elapsed / self.fps < self.DIRECTION_CONFIRM_SECONDS:
            return None
        return covariance / variance * self.fps

    def _direction(self) -> str | None:
        """'rising', 'falling', or None when speed cannot be called yet."""
        speed = self._vertical_speed()
        if speed is None:
            return None
        if speed <= -self.DIRECTION_MIN_SPEED_PX_PER_SECOND:
            return "rising"
        if speed >= self.DIRECTION_MIN_SPEED_PX_PER_SECOND:
            return "falling"
        return None

    def _emit(self, kind: str, frame: int, player_id: int | None, ball_id: int | None, score: float, shot_id: int | None = None) -> None:
        self.events.append({
            "event_id": len(self.events) + 1, "type": kind, "frame": frame,
            "timestamp_seconds": frame / self.fps, "player_track_id": player_id,
            "ball_track_id": ball_id, "shot_id": shot_id, "heuristic_score": score,
        })
        labels = {"shot_attempt_candidate": "SHOT CANDIDATE", "made_shot_candidate": "MADE CANDIDATE",
                  "missed_shot_candidate": "MISSED CANDIDATE", "shot_outcome_unknown": "SHOT OUTCOME UNKNOWN"}
        if kind in labels:
            self._last_overlay = (frame, labels[kind])

    def overlay(self, frame_index: int) -> str | None:
        if self._last_overlay is None or frame_index - self._last_overlay[0] >= self.overlay_frames:
            return None
        return self._last_overlay[1]

    def _finish_shot(self, outcome: str, frame: int) -> None:
        assert self._active is not None
        report = self._active.report
        report["outcome"], report["outcome_frame"] = outcome, frame
        kinds = {"made_candidate": ("made_shot_candidate", self.MADE_SCORE),
                 "missed_candidate": ("missed_shot_candidate", self.MISSED_SCORE),
                 "unknown": ("shot_outcome_unknown", 0.0)}
        kind, score = kinds[outcome]
        report["outcome_heuristic_score"] = score
        self._emit(kind, frame, report["player_track_id"], report["ball_track_id"], score, report["shot_id"])
        self._active, self._holder, self._release = None, None, None
        self._cooldown_until = frame + self.cooldown_frames
        self.state = "IDLE"

    def _resolve_pass_step(
        self, shot: _Shot, aligned: bool, previous: tuple[float, float] | None,
        x: float, y: float, direction: str | None,
    ) -> bool:
        """Advance the rim pass state one visible step; return a confirmed crossing.

        A crossing is confirmed only by an observed downward walk through the
        ROI: the ball is seen above the top edge, then seen below the bottom
        edge, with both boundaries crossed inside the rim width. A boundary that
        is assumed rather than observed (a sideways exit or a gap) clears the
        partial crossing instead of completing it, because the ball may have
        travelled around the rim rather than through it.
        """
        roi = self.rim_roi
        assert roi is not None
        descending = direction == "falling"
        previous_y = None if previous is None else previous[1]
        if previous is None:
            # First observation of an attempt, or the first after a gap. A gap
            # cannot complete a passage that was not observed, so the partial
            # crossing restarts from wherever the ball is now.
            if not aligned:
                shot.pass_state = "unknown"
            elif y < roi.y1:
                shot.pass_state = "above" if shot.above_seen else "unknown"
            elif y > roi.y2:
                shot.pass_state = "below"
            else:
                shot.pass_state = "unknown"
            return False
        if not aligned:
            # "above" and "below" describe which side of the rim the ball is on
            # and stay meaningful outside the rim width. "inside" does not: an
            # unaligned ball is no longer inside the rim opening, so a partial
            # crossing is dropped and the ball must be seen above the rim again
            # before any later downward step can complete a passage.
            if shot.pass_state == "inside":
                shot.pass_state = "unknown"
            return False
        if y < roi.y1:
            if shot.above_seen:
                shot.pass_state = "above"
            return False
        if y <= roi.y2:
            # The ball was seen above the top edge and has now descended into
            # the rim band. Entering the band some other way is not evidence.
            if shot.pass_state == "above" and descending:
                shot.pass_state = "inside"
            return False
        # y > roi.y2: below the bottom edge.
        if shot.pass_state == "below":
            return False
        # The passage is complete only if the ball actually left through the
        # bottom of the rim opening. Interpolating where the segment crosses the
        # bottom edge keeps sparse sampling honest: a ball that left through the
        # side while still inside the band never completes a passage.
        previous_x = previous[0]
        bottom_x = x if y == previous_y else previous_x + (x - previous_x) * (roi.y2 - previous_y) / (y - previous_y)
        crossed = (
            (shot.pass_state == "inside" or shot.pass_state == "above")
            and descending
            and roi.x1 <= bottom_x <= roi.x2
        )
        shot.pass_state = "below"
        return crossed

    def _observe_rim(self, ball: BallTrack, previous: BallTrack | None, direction: str | None, frame: int) -> None:
        assert self._active is not None and self.rim_roi is not None
        shot, roi = self._active, self.rim_roi
        x, y = ball["center"]
        aligned = roi.x1 <= x <= roi.x2
        if aligned and y < roi.y1:
            shot.above_seen = True
        crossed = self._resolve_pass_step(
            shot, aligned, None if previous is None else previous["center"], x, y, direction,
        )
        if crossed:
            shot.rim_crossed = True
        descending = direction == "falling"
        below_rim = descending and aligned and y > roi.y2
        # A made candidate needs the confirmed crossing plus one more visible
        # falling observation below the rim. The crossing step is itself the
        # first such observation, so a resolved passage resolves immediately,
        # while a crossing carried over from an earlier pass cannot emit one.
        if crossed:
            shot.below_frames = 1 if below_rim else 0
        elif below_rim and shot.rim_crossed:
            shot.below_frames += 1
        else:
            shot.below_frames = 0
        # A miss needs a visible falling observation below and outside the rim
        # width after the ball was seen above it, and no confirmed crossing.
        shot.outside_frames = (
            shot.outside_frames + 1
            if descending and not aligned and y > roi.y2 and shot.above_seen and not shot.rim_crossed
            else 0
        )
        if shot.below_frames >= self.MIN_OUTCOME_FRAMES:
            self._finish_shot("made_candidate", frame)
        elif shot.outside_frames >= self.MIN_OUTCOME_FRAMES:
            self._finish_shot("missed_candidate", frame)

    def update(
        self, ball: BallTrack | None, proxy: PossessionProxy | None,
        players: list[TrackedPerson], frame_index: int,
    ) -> list[Event]:
        """Consume visible evidence in frame order and return only newly emitted events."""
        start = len(self.events)
        proxy_id = None if proxy is None else proxy["player_track_id"]
        if proxy_id != self._proxy_id:
            self._emit("possession_proxy_change", frame_index, proxy_id,
                       None if ball is None else ball["ball_track_id"], 0.0 if proxy is None else proxy["heuristic_score"])
        self._proxy_id = proxy_id
        previous = self._previous_ball
        if ball is None or previous is None or self._previous_frame != frame_index - 1 or previous["ball_track_id"] != ball["ball_track_id"]:
            previous = None
            # A missing observation ends motion evidence; no position is
            # fabricated to bridge the gap.
            self._observations.clear()
        else:
            self._observe_motion(ball, frame_index)
        direction = self._direction()
        self._previous_ball, self._previous_frame = ball, frame_index
        if self.rim_roi is None:
            return self.events[start:]
        if previous is None and self._release is not None and self._active is None:
            self.state = "RELEASED"

        if self._active is not None:
            shot = self._active
            if (ball is None and frame_index - shot.last_seen > self.max_missing_frames
                    or ball is not None and ball["ball_track_id"] != shot.report["ball_track_id"]
                    or frame_index - shot.report["decision_frame"] > self.timeout_frames):
                self._finish_shot("unknown", frame_index)
            elif ball is not None:
                shot.last_seen = frame_index
                self._observe_rim(ball, previous, direction, frame_index)
            return self.events[start:]

        if frame_index < self._cooldown_until or ball is None:
            return self.events[start:]
        player = next((person for person in players if person["track_id"] == proxy_id), None)
        if player is not None and proxy is not None:
            self._holder = _Holder(frame_index, proxy_id, ball["ball_track_id"], player["bbox"].copy())
            self._release = None
            self.state = "POSSESSED"
            return self.events[start:]
        if self._release is None and self._holder is not None and previous is not None:
            holder = self._holder
            age = frame_index - holder.frame
            if holder.ball_id == ball["ball_track_id"] and age <= math.ceil(self.fps * self.RELEASE_WINDOW_SECONDS):
                if normalized_box_distance(ball["center"], holder.bbox) > self.possession_threshold:
                    owner = holder.player_id if age <= math.ceil(self.fps * self.ATTRIBUTION_WINDOW_SECONDS) else None
                    self._release = _Release(frame_index, ball["ball_track_id"], ball["center"], owner, self.rim_roi.distance(ball["center"]))
                    # Restart the motion window at release: the flight starts
                    # here, and the release observation itself is its first
                    # sample, so a time-based direction needs no extra latency.
                    self._observations.clear()
                    self._observe_motion(ball, frame_index)
                    self.state = "RELEASED"
                    return self.events[start:]
        if self._release is not None:
            release = self._release
            if frame_index - release.frame > self.timeout_frames or ball["ball_track_id"] != release.ball_id:
                self._release, self._holder = None, None
                self.state = "IDLE"
                return self.events[start:]
            rising = direction == "rising"
            if rising or self.state == "ASCENDING":
                self.state = "ASCENDING"
            roi = self.rim_roi
            x, y = ball["center"]
            approaching = (roi.x1 - roi.width <= x <= roi.x2 + roi.width
                           and y <= roi.y2 + 2 * roi.height
                           and y < release.center[1] - roi.height / 2
                           and roi.distance(ball["center"]) < release.rim_distance)
            if self.state == "ASCENDING" and approaching and previous is not None:
                report: ShotCandidate = {"shot_id": len(self.shots) + 1, "start_frame": release.frame,
                          "decision_frame": frame_index, "player_track_id": release.player_id,
                          "ball_track_id": release.ball_id, "outcome": "unknown", "outcome_frame": None,
                          "heuristic_score": self.ATTEMPT_SCORE, "outcome_heuristic_score": 0.0}
                self.shots.append(report)
                self._active = _Shot(report, frame_index)
                self.state = "SHOT_CANDIDATE"
                self._emit("shot_attempt_candidate", frame_index, release.player_id, release.ball_id, self.ATTEMPT_SCORE, report["shot_id"])
                self._observe_rim(ball, previous, direction, frame_index)
        return self.events[start:]

    def finalize(self, last_frame: int) -> None:
        """An unfinished visible attempt remains unknown at the end of the clip."""
        if self._active is not None:
            self._finish_shot("unknown", last_frame)

    def summary(self) -> dict:
        return {
            "enabled": self.rim_roi is not None, "attempts": len(self.shots),
            "made_candidates": sum(shot["outcome"] == "made_candidate" for shot in self.shots),
            "missed_candidates": sum(shot["outcome"] == "missed_candidate" for shot in self.shots),
            "unknown_outcomes": sum(shot["outcome"] == "unknown" for shot in self.shots),
            "shots": self.shots,
        }
