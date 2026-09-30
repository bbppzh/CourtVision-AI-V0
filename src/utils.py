"""Small helpers for input validation and frame annotation."""

from pathlib import Path

import cv2
import numpy as np

from src.detector import Detection


def validate_input_path(path: Path) -> None:
    """Fail early when the input is missing or is not a file."""
    if not path.is_file():
        raise ValueError(f"Input video does not exist or is not a file: {path}")


def validate_video_paths(input_path: Path, output_path: Path) -> None:
    """Validate video paths before loading a model or creating output files."""
    validate_input_path(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output path must end in .mp4 (the MP4 codec is used).")
    if output_path.is_dir():
        raise ValueError(f"Output path is a directory; provide an MP4 filename: {output_path}")


def draw_detection(frame: np.ndarray, detection: Detection) -> None:
    """Draw a player box and a readable confidence label in place."""
    x1, y1, x2, y2 = detection["bbox"]
    color = (0, 220, 80)  # OpenCV uses BGR.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"Player {detection['confidence']:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    label_x = max(0, min(x1, frame.shape[1] - text_width - 8))
    label_y = max(y1, text_height + baseline + 6)
    cv2.rectangle(
        frame,
        (label_x, label_y - text_height - baseline - 6),
        (label_x + text_width + 8, label_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (label_x + 4, label_y - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
