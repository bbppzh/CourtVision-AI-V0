"""Run a COCO person detector on individual video frames."""

from typing import TypedDict

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


PERSON_CLASS_ID = 0  # COCO's class index for "person".


class Detection(TypedDict):
    """One person detection in pixel coordinates."""

    bbox: list[int]
    confidence: float
    class_id: int
    class_name: str


class PlayerDetector:
    """Load YOLO once and detect people in one frame at a time."""

    def __init__(self, model_path: str = "yolo26n.pt", confidence: float = 0.5) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        self.confidence = confidence
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"Could not load YOLO model '{model_path}': {exc}") from exc

        if self.model.task != "detect":
            raise ValueError("The model must be an object detection model.")
        if self.model.names.get(PERSON_CLASS_ID) != "person":
            raise ValueError("Use COCO detection weights with class 0 named 'person'.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return only COCO person detections above the confidence threshold."""
        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )
        return self.filter_person_detections(results[0], self.confidence)

    @staticmethod
    def filter_person_detections(result: Results, confidence: float) -> list[Detection]:
        """Convert YOLO boxes to plain Python values and enforce person filtering."""
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            if class_id != PERSON_CLASS_ID or score < confidence:
                continue

            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": score,
                    "class_id": PERSON_CLASS_ID,
                    "class_name": "person",
                }
            )
        return detections
