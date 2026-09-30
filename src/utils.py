"""Small helpers for input validation and frame annotation."""

import colorsys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.detector import Detection

if TYPE_CHECKING:
    from src.tracker import TrackedPerson


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


def tracking_output_paths(
    output_path: Path, analytics_path: Path | None = None, heatmap_path: Path | None = None,
) -> tuple[Path, Path]:
    """Derive V1 sidecar names from the annotated video's filename."""
    return (
        analytics_path if analytics_path is not None else output_path.with_name(f"{output_path.stem}_tracking.json"),
        heatmap_path if heatmap_path is not None else output_path.with_name(f"{output_path.stem}_heatmap.png"),
    )


def validate_tracking_paths(
    input_path: Path, output_path: Path, analytics_path: Path, heatmap_path: Path,
) -> None:
    """Reject output collisions and invalid extensions before any model work."""
    validate_video_paths(input_path, output_path)
    for path, extension in ((analytics_path, ".json"), (heatmap_path, ".png")):
        if path.suffix.lower() != extension:
            raise ValueError(f"Output path must end in {extension}: {path}")
        if path.is_dir():
            raise ValueError(f"Output path is a directory: {path}")
    paths = [input_path, output_path, analytics_path, heatmap_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input, video, analytics, and heatmap paths must all be different.")


def track_color(track_id: int) -> tuple[int, int, int]:
    """Give each ID a repeatable bright BGR color without random state."""
    hue = (track_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_detection(frame: np.ndarray, detection: Detection) -> None:
    """Draw a player box and a readable confidence label in place."""
    x1, y1, x2, y2 = detection["bbox"]
    color = (0, 220, 80)  # OpenCV uses BGR.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_label(frame, f"Player {detection['confidence']:.2f}", x1, y1, color)


def draw_tracked_person(
    frame: np.ndarray, person: "TrackedPerson", points: list[tuple[float, float]],
    speed: float | None = None,
) -> None:
    """Draw a bounded trail, box, and ID label, optionally including px/s."""
    color = track_color(person["track_id"])
    if len(points) >= 2:
        polyline = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [polyline], False, color, 2, cv2.LINE_AA)
    x1, y1, x2, y2 = person["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"Player {person['track_id']} | {person['confidence']:.2f}"
    if speed is not None:
        label += f" | {speed:.0f} px/s"
    draw_label(frame, label, x1, y1, color)


def draw_label(
    frame: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int],
) -> None:
    """Fit a readable label inside the image, including on small demo frames."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    if text_width + 8 > frame.shape[1]:
        font_scale *= max(1, frame.shape[1] - 8) / (text_width + 8)
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = max(0, min(x, frame.shape[1] - text_width - 8))
    label_y = min(frame.shape[0] - baseline - 1, max(y, text_height + baseline + 6))
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
