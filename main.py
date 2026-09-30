"""Command-line entry point for CourtVision AI V0."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2

from src.detector import PlayerDetector
from src.utils import validate_video_paths
from src.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    """Read command-line options and reject invalid confidence values."""
    parser = argparse.ArgumentParser(description="Detect people in a basketball video.")
    parser.add_argument("--input", type=Path, required=True, help="Local input video")
    parser.add_argument("--output", type=Path, required=True, help="Annotated output .mp4")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum score (0 to 1)")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO COCO weights or local path")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    return args


def main() -> int:
    """Validate paths, load the model, and process the video."""
    args = parse_args()
    try:
        validate_video_paths(args.input, args.output)
        detector = PlayerDetector(model_path=args.model, confidence=args.confidence)
        print(f"Device: {detector.device}")
        started_at = perf_counter()
        processed = VideoProcessor(detector).process(args.input, args.output)
        elapsed = perf_counter() - started_at
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nProcessing complete.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Frames processed: {processed}")
    print(f"Processing time: {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
