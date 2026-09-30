"""Command-line entry point for CourtVision AI V1."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.utils import tracking_output_paths, validate_tracking_paths
from src.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    """Read command-line options and reject invalid confidence values."""
    parser = argparse.ArgumentParser(description="Track people and visualize pixel movement in a basketball video.")
    parser.add_argument("--input", type=Path, required=True, help="Local input video")
    parser.add_argument("--output", type=Path, required=True, help="Annotated output .mp4")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum score (0 to 1)")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO COCO weights or local path")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Packaged ByteTrack defaults or local YAML")
    parser.add_argument("--trajectory-length", type=int, default=40, help="Maximum recent center points per trail")
    parser.add_argument("--show-speed", action="store_true", help="Show smoothed image speed in px/s")
    parser.add_argument("--analytics-output", type=Path, help="JSON path; defaults to <output_stem>_tracking.json")
    parser.add_argument("--heatmap-output", type=Path, help="PNG path; defaults to <output_stem>_heatmap.png")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.trajectory_length <= 0:
        parser.error("--trajectory-length must be a positive integer")
    return args


def main() -> int:
    """Validate paths, load the model, and process the video."""
    args = parse_args()
    try:
        analytics_path, heatmap_path = tracking_output_paths(
            args.output, getattr(args, "analytics_output", None), getattr(args, "heatmap_output", None),
        )
        validate_tracking_paths(args.input, args.output, analytics_path, heatmap_path)
        detector = PlayerDetector(model_path=args.model, confidence=args.confidence)
        tracker = PlayerTracker(detector, getattr(args, "tracker", "bytetrack.yaml"))
        processor = VideoProcessor(
            detector, tracker=tracker,
            trajectory_length=getattr(args, "trajectory_length", 40),
            show_speed=getattr(args, "show_speed", False),
            analytics_output=analytics_path, heatmap_output=heatmap_path,
        )
        print(f"Device: {detector.device}")
        print("Tracker: ByteTrack")
        started_at = perf_counter()
        processed = processor.process(args.input, args.output)
        elapsed = perf_counter() - started_at
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nProcessing complete.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Tracking analytics: {analytics_path}")
    print(f"Movement heatmap: {heatmap_path}")
    print(f"Frames processed: {processed}")
    print(f"Unique tracks: {processor.unique_tracks}")
    print(f"Processing time: {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
