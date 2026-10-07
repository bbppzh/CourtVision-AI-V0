"""Accumulate tracked center visits in image coordinates."""

import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from src.tracker import TrackedPerson


class MovementHeatmap:
    """Keep one fixed-size grid; one visible center adds one frame of activity."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Heatmap width and height must be positive.")
        self.activity = np.zeros((height, width), dtype=np.float32)

    def update(self, tracks: list[TrackedPerson]) -> None:
        """Count each in-frame center once; ignore centers outside the image."""
        height, width = self.activity.shape
        for track in tracks:
            x, y = track["center"]
            if math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height:
                self.activity[int(y), int(x)] += 1

    def render(self) -> np.ndarray:
        """Return a normalized color PNG image, with zero activity shown black."""
        if not np.any(self.activity):
            return np.zeros((*self.activity.shape, 3), dtype=np.uint8)
        # Blur only when exporting; the grid still stores exact center visits.
        blurred = cv2.GaussianBlur(self.activity, (0, 0), sigmaX=8, sigmaY=8)
        scaled = np.clip(blurred / blurred.max() * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        colored[scaled == 0] = 0
        return colored

    def save(self, path: Path) -> None:
        """Save a PNG atomically so an export error preserves previous output."""
        if path.suffix.lower() != ".png":
            raise ValueError("Heatmap output path must end in .png.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".png", delete=False) as file:
            temporary_path = Path(file.name)
        try:
            if not cv2.imwrite(str(temporary_path), self.render()):
                raise RuntimeError(f"Could not write heatmap: {path}")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
