"""Read, annotate, and write a video frame by frame."""

import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from src.detector import PlayerDetector
from src.analytics import MotionAnalytics
from src.heatmap import MovementHeatmap
from src.track_history import TrackHistory
from src.tracker import PlayerTracker
from src.utils import (
    draw_detection, draw_tracked_person, tracking_output_paths,
    validate_tracking_paths, validate_video_paths,
)


class VideoProcessor:
    """Process a local video while preserving its dimensions and frame rate."""

    def __init__(
        self, detector: PlayerDetector, tracker: PlayerTracker | None = None,
        trajectory_length: int = 40, show_speed: bool = False,
        analytics_output: Path | None = None, heatmap_output: Path | None = None,
    ) -> None:
        if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
            raise ValueError("Trajectory length must be a positive integer.")
        self.detector = detector
        self.tracker = tracker
        self.trajectory_length = trajectory_length
        self.show_speed = show_speed
        self.analytics_output = analytics_output
        self.heatmap_output = heatmap_output
        self._requested_analytics_output = analytics_output
        self._requested_heatmap_output = heatmap_output
        self.history: TrackHistory | None = None
        self.analytics: MotionAnalytics | None = None
        self.heatmap: MovementHeatmap | None = None

    @property
    def unique_tracks(self) -> int:
        """Count tracker identities observed above the confidence threshold."""
        return self.analytics.unique_tracks if self.analytics is not None else 0

    def _start_tracking(self, width: int, height: int, fps: float) -> None:
        """Reset per-video tracking, bounded trails, and motion summaries."""
        if self.tracker is not None:
            self.tracker.start()
            self.history = TrackHistory(self.trajectory_length)
            self.analytics = MotionAnalytics(width, height, fps)
            self.heatmap = MovementHeatmap(width, height)

    def _annotate_frame(self, frame: np.ndarray, frame_index: int) -> int:
        """Update frame analytics before drawing; return the visible person count."""
        if self.tracker is None:
            detections = self.detector.detect(frame)
            for detection in detections:
                draw_detection(frame, detection)
            return len(detections)
        tracks = self.tracker.track(frame)
        assert self.history is not None and self.analytics is not None and self.heatmap is not None
        self.history.update(tracks, frame_index)
        self.analytics.update(tracks, frame_index)
        self.heatmap.update(tracks)
        for person in tracks:
            speed = self.analytics.speed(person["track_id"]) if self.show_speed else None
            draw_tracked_person(frame, person, self.history.get_points(person["track_id"]), speed)
        return len(tracks)

    @staticmethod
    def _temporary_path(destination: Path) -> Path:
        """Stage each output in its destination folder for a final rename."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.stem}-",
            suffix=destination.suffix, delete=False,
        ) as file:
            return Path(file.name)

    @staticmethod
    def _publish_tracking_outputs(outputs: list[tuple[Path, Path]]) -> None:
        """Publish the three finished files, restoring old files on rename errors."""
        backups: dict[Path, Path | None] = {}
        published: set[Path] = set()
        try:
            for staged_path, destination in outputs:
                backup = None
                if destination.exists():
                    backup = VideoProcessor._temporary_path(destination)
                    try:
                        destination.replace(backup)
                    except OSError:
                        backup.unlink(missing_ok=True)
                        raise
                backups[destination] = backup
                staged_path.replace(destination)
                published.add(destination)
        except OSError as exc:
            failed_restores: list[Path] = []
            for destination, backup in reversed(list(backups.items())):
                try:
                    if backup is not None:
                        backup.replace(destination)
                    elif destination in published:
                        destination.unlink(missing_ok=True)
                except OSError:
                    if backup is not None:
                        failed_restores.append(backup)
            if failed_restores:
                raise RuntimeError(
                    "Output publication failed and some previous files could not be "
                    f"restored. Recoverable backups: {failed_restores}"
                ) from exc
            raise
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)

    def process(self, input_path: Path, output_path: Path) -> int:
        """Write annotated MP4 frames and return the number processed."""
        validate_video_paths(input_path, output_path)
        if self.tracker is not None:
            self.analytics_output, self.heatmap_output = tracking_output_paths(
                output_path, self._requested_analytics_output, self._requested_heatmap_output,
            )
            validate_tracking_paths(input_path, output_path, self.analytics_output, self.heatmap_output)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open input video: {input_path}")

        writer = None
        temporary_output = None
        staged_exports: list[tuple[Path, Path]] = []
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise ValueError("Input video has invalid width or height.")
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Input video has invalid FPS.")
            if width % 2 or height % 2:
                raise ValueError(
                    f"Input resolution is {width}x{height}. The MP4 codec requires "
                    "even width and height to preserve the original resolution."
                )

            # Finish the video before replacing the requested output file.
            temporary_output = self._temporary_path(output_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temporary_output), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")
            self._start_tracking(width, height, fps)

            print(f"Video: {width}x{height} at {fps:.2f} FPS")
            print(f"Frames reported: {total_frames if total_frames > 0 else 'unknown'}")
            frames_processed = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError("A decoded frame does not match the video's resolution.")
                visible_count = self._annotate_frame(frame, frames_processed)
                writer.write(frame)
                frames_processed += 1
                if frames_processed == 1 or frames_processed % 30 == 0:
                    total = total_frames if total_frames > 0 else "?"
                    details = (
                        f"Active tracks: {visible_count} | Unique tracks: {self.unique_tracks}"
                        if self.tracker is not None else f"Players detected: {visible_count}"
                    )
                    print(f"Processing: {frames_processed} / {total} frames | {details}")

            if frames_processed == 0:
                raise ValueError("Input video contains no readable frames.")
            if total_frames > 0 and frames_processed < total_frames:
                raise ValueError(
                    f"Video decoding stopped after {frames_processed} of "
                    f"{total_frames} reported frames. The input may be corrupted "
                    "or its frame count may be inaccurate."
                )
            writer.release()
            writer = None
            if self.tracker is not None:
                assert self.analytics is not None and self.heatmap is not None
                assert self.analytics_output is not None and self.heatmap_output is not None
                for destination in (self.analytics_output, self.heatmap_output):
                    staged_exports.append((self._temporary_path(destination), destination))
                self.analytics.save_json(staged_exports[0][0], frames_processed)
                self.heatmap.save(staged_exports[1][0])
            if self.tracker is not None:
                self._publish_tracking_outputs([(temporary_output, output_path), *staged_exports])
            else:
                temporary_output.replace(output_path)
            temporary_output = None
            return frames_processed
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if temporary_output is not None:
                temporary_output.unlink(missing_ok=True)
            for staged_path, _ in staged_exports:
                staged_path.unlink(missing_ok=True)
