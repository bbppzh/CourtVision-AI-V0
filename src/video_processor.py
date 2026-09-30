"""Read, annotate, and write a video frame by frame."""

import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2

from src.detector import PlayerDetector
from src.utils import draw_detection, validate_video_paths


class VideoProcessor:
    """Process a local video while preserving its dimensions and frame rate."""

    def __init__(self, detector: PlayerDetector) -> None:
        self.detector = detector

    def process(self, input_path: Path, output_path: Path) -> int:
        """Write annotated MP4 frames and return the number processed."""
        validate_video_paths(input_path, output_path)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open input video: {input_path}")

        writer = None
        temporary_output = None
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

            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Finish the video before replacing the requested output file.
            with NamedTemporaryFile(
                dir=output_path.parent, prefix=f".{output_path.stem}-",
                suffix=".mp4", delete=False,
            ) as temporary_file:
                temporary_output = Path(temporary_file.name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temporary_output), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")

            print(f"Video: {width}x{height} at {fps:.2f} FPS")
            print(f"Frames reported: {total_frames if total_frames > 0 else 'unknown'}")
            frames_processed = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError("A decoded frame does not match the video's resolution.")
                detections = self.detector.detect(frame)
                for detection in detections:
                    draw_detection(frame, detection)
                writer.write(frame)
                frames_processed += 1
                if frames_processed == 1 or frames_processed % 30 == 0:
                    total = total_frames if total_frames > 0 else "?"
                    print(
                        f"Processing: {frames_processed} / {total} frames | "
                        f"Players detected: {len(detections)}"
                    )

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
            temporary_output.replace(output_path)
            temporary_output = None
            return frames_processed
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if temporary_output is not None:
                temporary_output.unlink(missing_ok=True)
