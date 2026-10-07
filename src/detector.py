"""Run COCO person detection, with shared sports-ball inference for V2."""

import math
from typing import TypedDict

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results


PERSON_CLASS_ID = 0  # COCO's class index for "person".


class Detection(TypedDict):
    """One person detection in pixel coordinates."""

    bbox: list[int]
    confidence: float
    class_id: int
    class_name: str


class BallDetection(Detection):
    """A generic COCO sports-ball observation, not a basketball identity."""

    center: tuple[float, float]


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
        self.sports_ball_class_id = next(
            (class_id for class_id, name in self.model.names.items() if name == "sports ball"), None,
        )

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

    def detect_scene(
        self, frame: np.ndarray, person_inference_confidence: float,
        ball_confidence: float = 0.25,
    ) -> tuple[Boxes | None, list[BallDetection]]:
        """Use one YOLO pass for weak person boxes and visible sports-ball boxes."""
        if not 0 <= ball_confidence <= 1:
            raise ValueError("Ball confidence must be between 0 and 1.")
        if self.sports_ball_class_id is None:
            raise ValueError("V2 needs COCO weights containing the 'sports ball' class.")
        with torch.inference_mode():
            result = self.model.predict(
                source=frame, classes=[PERSON_CLASS_ID, self.sports_ball_class_id],
                conf=min(self.confidence, person_inference_confidence, ball_confidence),
                device=self.device, verbose=False,
            )[0]
        people = None if result.boxes is None else result.boxes[result.boxes.cls == PERSON_CLASS_ID]
        balls = self.filter_sports_ball_detections(result, ball_confidence, self.sports_ball_class_id)
        return people, balls

    @staticmethod
    def filter_sports_ball_detections(
        result: Results, confidence: float, class_id: int,
    ) -> list[BallDetection]:
        """Keep finite sports-ball boxes above threshold, preserving float centers."""
        balls: list[BallDetection] = []
        if result.boxes is None:
            return balls
        for box in result.boxes:
            score = float(box.conf.item())
            values = box.xyxy[0].tolist()
            if float(box.cls.item()) != class_id or not math.isfinite(score) or score < confidence:
                continue
            if not all(math.isfinite(value) for value in values):
                continue
            x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1:
                continue
            balls.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": score,
                "class_id": class_id, "class_name": "sports ball",
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            })
        return balls

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
