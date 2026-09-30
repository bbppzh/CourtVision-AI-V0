"""Associate YOLO person detections using Ultralytics' official ByteTrack."""

import math
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Boxes, Results
from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

from src.detector import Detection, PERSON_CLASS_ID, PlayerDetector


class TrackedPerson(Detection):
    """A temporary tracker identity and its bounding-box center in pixels."""

    track_id: int
    center: tuple[float, float]


def load_tracker_config(config_name: str) -> IterableSimpleNamespace:
    """Load the packaged ByteTrack defaults or a local ByteTrack YAML file."""
    config_path = (
        ROOT / "cfg" / "trackers" / "bytetrack.yaml"
        if config_name == "bytetrack.yaml" else Path(config_name)
    )
    if not config_path.is_file() or config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"ByteTrack configuration must be an existing YAML file: {config_name}")
    try:
        settings = YAML.load(config_path)
    except Exception as exc:
        raise ValueError(f"Could not read ByteTrack configuration: {config_name}") from exc
    if not isinstance(settings, dict) or settings.get("tracker_type") != "bytetrack":
        raise ValueError("The tracker configuration must use tracker_type: bytetrack.")
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"ByteTrack {key} must be a number between 0 and 1.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"ByteTrack {key} must be a number between 0 and 1.")
    if settings["track_low_thresh"] >= settings["track_high_thresh"]:
        raise ValueError("ByteTrack track_low_thresh must be below track_high_thresh.")
    buffer = settings.get("track_buffer")
    if isinstance(buffer, bool) or not isinstance(buffer, int) or buffer <= 0:
        raise ValueError("ByteTrack track_buffer must be a positive integer.")
    if not isinstance(settings.get("fuse_score"), bool):
        raise ValueError("ByteTrack fuse_score must be true or false.")
    return IterableSimpleNamespace(**settings)


class PlayerTracker:
    """Run person inference once per frame, then let ByteTrack assign IDs."""

    def __init__(self, detector: PlayerDetector, tracker_config: str = "bytetrack.yaml") -> None:
        self.detector = detector
        self.config = load_tracker_config(tracker_config)
        self._tracker = None

    def start(self) -> None:
        """Create fresh tracker state for each video, including a fresh ID counter."""
        # Imported here so detection-only use does not load tracking dependencies.
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(self.config)

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        """Return visible people with IDs; retain weak detections for association."""
        if self._tracker is None:
            raise RuntimeError("Start the tracker before processing a video.")
        with torch.inference_mode():
            results = self.detector.model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                # ByteTrack needs weak detections to reconnect existing tracks.
                # Apply the user's display/analytics threshold after association.
                conf=min(self.detector.confidence, self.config.track_low_thresh),
                device=self.detector.device,
                verbose=False,
            )
        boxes = results[0].boxes
        if boxes is None:
            boxes = Boxes(np.empty((0, 6), dtype=np.float32), frame.shape[:2])
        tracked_rows = self._tracker.update(boxes.cpu().numpy(), frame)
        if len(tracked_rows) == 0:
            return []
        # Official ByteTrack rows end with the original detection index.
        # Results expects xyxy, track_id, confidence, class_id (without that index).
        tracked_result = Results(
            orig_img=frame, path="", names=self.detector.model.names,
            boxes=tracked_rows[:, :-1],
        )
        return self.extract_tracked_people(tracked_result, self.detector.confidence)

    @staticmethod
    def extract_tracked_people(result: Results, confidence: float) -> list[TrackedPerson]:
        """Convert tracked boxes to Python values and reject missing/invalid IDs."""
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        people: list[TrackedPerson] = []
        for box in boxes:
            if box.id is None:
                continue
            class_value = float(box.cls.item())
            score = float(box.conf.item())
            track_value = float(box.id.item())
            coordinates = box.xyxy[0].tolist()
            if not all(math.isfinite(value) for value in [class_value, score, track_value, *coordinates]):
                continue
            if class_value != PERSON_CLASS_ID or score < confidence:
                continue
            if track_value < 0 or not track_value.is_integer():
                continue
            x1, y1, x2, y2 = coordinates
            if x2 <= x1 or y2 <= y1:
                continue
            people.append({
                "track_id": int(track_value),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": score,
                "class_id": PERSON_CLASS_ID,
                "class_name": "person",
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            })
        return people
