"""Measure image-plane player motion using only consecutive observations."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from src.tracker import TrackedPerson
    from src.ball_tracker import BallTrack
    from src.interaction import PossessionProxy


@dataclass
class _TrackStats:
    """Small running summary; no complete trajectory is stored here."""

    frames_seen: int = 0
    last_center: tuple[float, float] | None = None
    last_frame: int | None = None
    cumulative_distance: float = 0.0
    motion_intervals: int = 0
    speed_samples: deque[float] = field(default_factory=deque)
    max_speed: float = 0.0


class MotionAnalytics:
    """Maintain pixel motion summaries and a small speed window for each ID.

    Memory scales as O(unique track IDs * smoothing_window), independent of
    video length for a fixed set of IDs. Summaries keep all unique IDs so the
    final report can include players that have already left the frame.
    """

    def __init__(
        self, width: int, height: int, fps: float, smoothing_window: int = 5
    ) -> None:
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive.")
        if (
            isinstance(smoothing_window, bool)
            or not isinstance(smoothing_window, int)
            or smoothing_window <= 0
        ):
            raise ValueError("smoothing_window must be a positive integer.")

        self.width = width
        self.height = height
        self.fps = fps
        self.smoothing_window = smoothing_window
        self._frame_diagonal = math.hypot(width, height)
        self._tracks: dict[int, _TrackStats] = {}
        self._active_track_ids: set[int] = set()

    @property
    def unique_tracks(self) -> int:
        """Count every distinct tracking ID observed in the video."""
        return len(self._tracks)

    def update(self, tracks: Iterable[TrackedPerson], frame_index: int) -> None:
        """Measure displacement only when this ID was seen in the prior frame."""
        visible_ids: set[int] = set()
        for track in tracks:
            track_id = track["track_id"]
            visible_ids.add(track_id)
            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackStats(
                    speed_samples=deque(maxlen=self.smoothing_window)
                )
            self._update_track(self._tracks[track_id], track["center"], frame_index)

        # A missing observation ends the current speed window immediately.
        for track_id in self._active_track_ids - visible_ids:
            self._tracks[track_id].speed_samples.clear()
        self._active_track_ids = visible_ids

    def _update_track(
        self, stats: _TrackStats, center: tuple[float, float], frame_index: int
    ) -> None:
        if stats.last_center is not None and stats.last_frame == frame_index - 1:
            distance = math.hypot(
                center[0] - stats.last_center[0], center[1] - stats.last_center[1]
            )
            stats.cumulative_distance += distance
            stats.motion_intervals += 1
            stats.speed_samples.append(distance * self.fps)
            smoothed_speed = sum(stats.speed_samples) / len(stats.speed_samples)
            stats.max_speed = max(stats.max_speed, smoothed_speed)
        else:
            stats.speed_samples.clear()

        stats.frames_seen += 1
        stats.last_center = center
        stats.last_frame = frame_index

    def speed(self, track_id: int) -> float:
        """Return the recent average interval speed in pixels per second."""
        stats = self._tracks.get(track_id)
        if stats is None or not stats.speed_samples:
            return 0.0
        return sum(stats.speed_samples) / len(stats.speed_samples)

    def to_dict(self, frames_processed: int) -> dict:
        """Build a report in O(unique IDs), using image-space units throughout."""
        if (
            isinstance(frames_processed, bool)
            or not isinstance(frames_processed, int)
            or frames_processed < 0
        ):
            raise ValueError("frames_processed must be a nonnegative integer.")

        summaries = []
        for track_id, stats in sorted(self._tracks.items()):
            average_speed = 0.0
            if stats.motion_intervals:
                observed_seconds = stats.motion_intervals / self.fps
                average_speed = stats.cumulative_distance / observed_seconds
            summaries.append({
                "track_id": track_id,
                "frames_seen": stats.frames_seen,
                "cumulative_pixel_distance": stats.cumulative_distance,
                "average_pixel_speed": average_speed,
                "max_pixel_speed": stats.max_speed,
                "normalized_distance": stats.cumulative_distance / self._frame_diagonal,
            })

        return {
            "video": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "frames_processed": frames_processed,
            },
            "coordinate_system": "image bounding_box_center",
            "units": {"distance": "px", "speed": "px/s"},
            "scientific_warning": (
                "Pixel motion depends on camera perspective, zoom, and camera movement. "
                "These are not real-world distances or speeds. Camera calibration and "
                "perspective correction are needed to measure physical units. "
                "Normalized distance divides pixel distance by the frame diagonal."
            ),
            "tracks": summaries,
        }

    def save_json(self, path: Path, frames_processed: int, extra_fields: dict | None = None) -> None:
        """Replace the destination only after valid JSON is completely written."""
        report = self.to_dict(frames_processed)
        if extra_fields is not None:
            if report.keys() & extra_fields.keys():
                raise ValueError("Additional analytics fields must not overwrite V1 fields.")
            report.update(extra_fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}-", suffix=".tmp", delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(report, temporary_file, indent=2, allow_nan=False)
                temporary_file.write("\n")
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class BasketballAnalytics:
    """Running visible-ball and possession-proxy counts; never an accuracy score."""

    def __init__(self) -> None:
        self.frames_detected = 0
        self.unknown_frames = 0
        self.player_frames: dict[int, int] = {}
        self.ball_track_ids: set[int] = set()

    def update(self, ball: BallTrack | None, proxy: PossessionProxy | None) -> None:
        if ball is not None:
            self.frames_detected += 1
            self.ball_track_ids.add(ball["ball_track_id"])
        if proxy is None:
            self.unknown_frames += 1
        else:
            identity = proxy["player_track_id"]
            self.player_frames[identity] = self.player_frames.get(identity, 0) + 1

    def to_dict(self, frames_processed: int) -> dict:
        return {
            "basketball": {
                "model_class": "sports ball", "frames_detected": self.frames_detected,
                "detection_rate": self.frames_detected / frames_processed if frames_processed else 0.0,
                "primary_ball_tracks": len(self.ball_track_ids),
                "rate_definition": "Frames with a selected visible sports-ball candidate / frames processed; not precision, recall, or accuracy.",
            },
            "possession_proxy": {
                "player_frames": {str(identity): count for identity, count in sorted(self.player_frames.items())},
                "unknown_frames": self.unknown_frames,
                "warning": "Temporally confirmed image proximity does not prove physical possession.",
            },
        }
