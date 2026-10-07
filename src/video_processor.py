"""Read, annotate, and write a video frame by frame."""

import math
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import cv2
import numpy as np

from src.detector import PlayerDetector
from src.analytics import BasketballAnalytics, MotionAnalytics
from src.ball_tracker import BallTracker
from src.events import EventAnalyzer, RimROI
from src.interaction import PossessionEstimator
from src.heatmap import MovementHeatmap
from src.track_history import TrackHistory
from src.tracker import PlayerTracker
from src.utils import (
    draw_ball, draw_detection, draw_label, draw_possession_proxy, draw_rim_roi,
    draw_tracked_person, tracking_output_paths,
    validate_tracking_paths, validate_video_paths,
)


class VideoProcessor:
    """Process a local video while preserving its dimensions and frame rate."""

    def __init__(
        self, detector: PlayerDetector, tracker: PlayerTracker | None = None,
        trajectory_length: int = 40, show_speed: bool = False,
        analytics_output: Path | None = None, heatmap_output: Path | None = None,
        enable_basketball: bool = False, ball_confidence: float = 0.25,
        rim_roi: RimROI | None = None, possession_distance_threshold: float = 0.4,
        possession_min_frames: int = 3, show_events: bool = False,
        trace_ball_frames: bool = False,
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
        if not 0 <= ball_confidence <= 1:
            raise ValueError("Ball confidence must be between 0 and 1.")
        if enable_basketball and tracker is None:
            raise ValueError("V2 basketball analysis requires a player tracker.")
        # Validate possession settings even before any video/model inference.
        PossessionEstimator(possession_distance_threshold, possession_min_frames)
        self.enable_basketball = enable_basketball
        self.ball_confidence = ball_confidence
        self.rim_roi = rim_roi
        self.possession_distance_threshold = possession_distance_threshold
        self.possession_min_frames = possession_min_frames
        self.show_events = show_events
        self.trace_ball_frames = trace_ball_frames
        self.ball_tracker: BallTracker | None = None
        self.interaction: PossessionEstimator | None = None
        self.event_analyzer: EventAnalyzer | None = None
        self.basketball_analytics: BasketballAnalytics | None = None
        self.processing_seconds = 0.0
        self.processing_fps = 0.0
        self._ball_visible = False

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
            if self.enable_basketball:
                self.ball_tracker = BallTracker(width, height, self.trajectory_length, fps=fps)
                self.interaction = PossessionEstimator(self.possession_distance_threshold, self.possession_min_frames)
                self.event_analyzer = EventAnalyzer(fps, self.rim_roi, self.possession_distance_threshold)
                self.basketball_analytics = BasketballAnalytics(trace_ball_frames=self.trace_ball_frames)

    def _annotate_frame(self, frame: np.ndarray, frame_index: int) -> int:
        """Update frame analytics before drawing; return the visible person count."""
        if self.tracker is None:
            detections = self.detector.detect(frame)
            for detection in detections:
                draw_detection(frame, detection)
            return len(detections)
        ball, proxy = None, None
        if self.enable_basketball:
            boxes, detections = self.detector.detect_scene(frame, self.tracker.config.track_low_thresh, self.ball_confidence)
            tracks = self.tracker.track_detections(frame, boxes)
            assert self.ball_tracker is not None and self.interaction is not None
            assert self.event_analyzer is not None and self.basketball_analytics is not None
            ball = self.ball_tracker.update(detections, frame_index)
            proxy = self.interaction.update(ball, tracks, frame_index)
            self.event_analyzer.update(ball, proxy, tracks, frame_index)
            self.basketball_analytics.update(ball, proxy)
            self._ball_visible = ball is not None
        else:
            tracks = self.tracker.track(frame)
        assert self.history is not None and self.analytics is not None and self.heatmap is not None
        self.history.update(tracks, frame_index)
        self.analytics.update(tracks, frame_index)
        self.heatmap.update(tracks)
        for person in tracks:
            speed = self.analytics.speed(person["track_id"]) if self.show_speed else None
            draw_tracked_person(frame, person, self.history.get_points(person["track_id"]), speed)
        if self.enable_basketball:
            if self.rim_roi is not None:
                draw_rim_roi(frame, self.rim_roi)
            if ball is not None:
                draw_ball(frame, ball, list(self.ball_tracker.history))
            if self.show_events:
                if proxy is not None:
                    draw_possession_proxy(frame, proxy, tracks)
                label = self.event_analyzer.overlay(frame_index)
                if label is not None:
                    draw_label(frame, label, 3, 20, (0, 170, 255), font_scale=0.5)
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
                try:
                    backup.unlink(missing_ok=True)
                except OSError as exc:
                    # Publication has succeeded; leftover backups are a cleanup issue.
                    print(
                        f"Warning: Outputs were saved, but could not remove old "
                        f"backup '{backup}': {exc}",
                        file=sys.stderr,
                    )

    def process(self, input_path: Path, output_path: Path) -> int:
        """Write annotated MP4 frames and return the number processed."""
        validate_video_paths(input_path, output_path)
        started_at = perf_counter()
        self.processing_seconds = self.processing_fps = 0.0
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
            if self.rim_roi is not None:
                self.rim_roi.validate_dimensions(width, height)

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
                    if self.enable_basketball:
                        details += f" | Ball visible: {'yes' if self._ball_visible else 'no'}"
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
            self.processing_seconds = perf_counter() - started_at
            self.processing_fps = frames_processed / self.processing_seconds
            if self.tracker is not None:
                assert self.analytics is not None and self.heatmap is not None
                assert self.analytics_output is not None and self.heatmap_output is not None
                for destination in (self.analytics_output, self.heatmap_output):
                    staged_exports.append((self._temporary_path(destination), destination))
                if self.enable_basketball:
                    assert self.event_analyzer is not None and self.basketball_analytics is not None
                    self.event_analyzer.finalize(frames_processed - 1)
                    extra = self.basketball_analytics.to_dict(frames_processed)
                    extra.update({
                        "shot_candidates": self.event_analyzer.summary(), "events": self.event_analyzer.events,
                        "event_rules": {
                            "rim_roi": None if self.rim_roi is None else [self.rim_roi.x1, self.rim_roi.y1, self.rim_roi.x2, self.rim_roi.y2],
                            "ball_confidence": self.ball_confidence,
                            "possession_distance_threshold": self.possession_distance_threshold,
                            "possession_min_frames": self.possession_min_frames,
                            "score_definition": "Fixed rule-strength scores, not calibrated probabilities; events are candidates, not ground truth.",
                        },
                        "performance": {"source_fps": fps, "processing_seconds": self.processing_seconds,
                                        "processing_fps": self.processing_fps,
                                        "measurement": "Video processing through encoder close; excludes model loading and sidecar publication."},
                    })
                    self.analytics.save_json(staged_exports[0][0], frames_processed, extra_fields=extra)
                else:
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
