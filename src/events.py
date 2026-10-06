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


class EventAnalyzer:
    """POSSESSED → RELEASED → ASCENDING → SHOT_CANDIDATE → outcome.

    Scores are fixed rule-strength labels, not calibrated probabilities.
    Missing observations cannot establish a rim crossing or a made candidate.
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
    DIRECTION_EPSILON_PX = 1.0
    MIN_DIRECTION_FRAMES = 2
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
        self.events: list[Event] = []
        self.shots: list[ShotCandidate] = []
        self.state = "IDLE"
        self._holder: _Holder | None = None
        self._release: _Release | None = None
        self._active: _Shot | None = None
        self._previous_ball: BallTrack | None = None
        self._previous_frame: int | None = None
        self._vertical_steps: deque[float] = deque(maxlen=3)
        self._upward_frames = 0
        self._proxy_id: int | None = None
        self._cooldown_until = -1
        self._last_overlay: tuple[int, str] | None = None

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

    def _observe_rim(self, ball: BallTrack, previous: BallTrack | None, velocity_y: float | None, frame: int) -> None:
        assert self._active is not None and self.rim_roi is not None
        shot, roi = self._active, self.rim_roi
        x, y = ball["center"]
        if previous is None:
            shot.above_seen = shot.rim_crossed = False
            shot.below_frames = shot.outside_frames = 0
        aligned = roi.x1 <= x <= roi.x2
        if aligned and y < roi.y1:
            shot.above_seen = True
        if previous is not None and shot.above_seen:
            px, py = previous["center"]
            if py < roi.y1 <= y and y > py:
                crossing_x = px + (x - px) * (roi.y1 - py) / (y - py)
                bottom_x = x if y <= roi.y2 else px + (x - px) * (roi.y2 - py) / (y - py)
                if roi.x1 <= crossing_x <= roi.x2 and roi.x1 <= bottom_x <= roi.x2:
                    shot.rim_crossed = True
        descending = velocity_y is not None and velocity_y > self.DIRECTION_EPSILON_PX
        shot.below_frames = shot.below_frames + 1 if descending and aligned and y > roi.y2 and shot.rim_crossed else 0
        shot.outside_frames = shot.outside_frames + 1 if descending and not aligned and y > roi.y2 and shot.above_seen and not shot.rim_crossed else 0
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
            self._vertical_steps.clear()
            self._upward_frames = 0
        else:
            self._vertical_steps.append(ball["center"][1] - previous["center"][1])
        velocity_y = sum(self._vertical_steps) / len(self._vertical_steps) if self._vertical_steps else None
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
                self._observe_rim(ball, previous, velocity_y, frame_index)
            return self.events[start:]

        if frame_index < self._cooldown_until or ball is None:
            return self.events[start:]
        player = next((person for person in players if person["track_id"] == proxy_id), None)
        if player is not None and proxy is not None:
            self._holder = _Holder(frame_index, proxy_id, ball["ball_track_id"], player["bbox"].copy())
            self._release = None
            self._upward_frames = 0
            self.state = "POSSESSED"
            return self.events[start:]
        if self._release is None and self._holder is not None and previous is not None:
            holder = self._holder
            age = frame_index - holder.frame
            if holder.ball_id == ball["ball_track_id"] and age <= math.ceil(self.fps * self.RELEASE_WINDOW_SECONDS):
                if normalized_box_distance(ball["center"], holder.bbox) > self.possession_threshold:
                    owner = holder.player_id if age <= math.ceil(self.fps * self.ATTRIBUTION_WINDOW_SECONDS) else None
                    self._release = _Release(frame_index, ball["ball_track_id"], ball["center"], owner, self.rim_roi.distance(ball["center"]))
                    self._upward_frames = 0
                    self.state = "RELEASED"
                    return self.events[start:]
        if self._release is not None:
            release = self._release
            if frame_index - release.frame > self.timeout_frames or ball["ball_track_id"] != release.ball_id:
                self._release, self._holder = None, None
                self.state = "IDLE"
                return self.events[start:]
            rising = velocity_y is not None and velocity_y < -self.DIRECTION_EPSILON_PX
            self._upward_frames = self._upward_frames + 1 if rising else 0
            if self._upward_frames >= self.MIN_DIRECTION_FRAMES:
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
                self._observe_rim(ball, previous, velocity_y, frame_index)
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
