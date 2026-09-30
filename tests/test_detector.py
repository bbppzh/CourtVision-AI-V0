"""Fast tests that do not download model weights or need a real game video."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.detector import PlayerDetector
from src.video_processor import VideoProcessor


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_invalid_confidence_is_rejected(monkeypatch, confidence: float) -> None:
    monkeypatch.setattr("src.detector.YOLO", lambda _: object())
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        PlayerDetector(confidence=confidence)


def test_detection_filter_keeps_only_confident_people() -> None:
    def box(class_id: int, confidence: float, coordinates: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            cls=np.array([class_id]),
            conf=np.array([confidence]),
            xyxy=np.array([coordinates]),
        )

    result = SimpleNamespace(
        boxes=[
            box(0, 0.91, [10, 20, 30, 40]),
            box(0, 0.35, [1, 2, 3, 4]),
            box(32, 0.99, [5, 6, 7, 8]),
        ]
    )

    detections = PlayerDetector.filter_person_detections(result, confidence=0.5)

    assert detections == [
        {
            "bbox": [10, 20, 30, 40],
            "confidence": pytest.approx(0.91),
            "class_id": 0,
            "class_name": "person",
        }
    ]


@pytest.mark.parametrize("task,names,error", [
    ("classify", {0: "person"}, "object detection model"),
    ("detect", {0: "basketball"}, "class 0 named 'person'"),
])
def test_incompatible_model_is_rejected(monkeypatch, task, names, error) -> None:
    model = SimpleNamespace(task=task, names=names)
    monkeypatch.setattr("src.detector.YOLO", lambda _: model)

    with pytest.raises(ValueError, match=error):
        PlayerDetector("custom.pt")


@pytest.mark.parametrize("cuda_available,expected_device", [(True, "cuda"), (False, "cpu")])
def test_inference_selects_device_and_disables_gradients(monkeypatch, cuda_available, expected_device):
    import torch

    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def predict(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert kwargs["source"] is frame
        assert kwargs["classes"] == [0]
        assert kwargs["conf"] == 0.6
        assert kwargs["device"] == expected_device
        return [SimpleNamespace(boxes=None)]

    model = SimpleNamespace(task="detect", names={0: "person"}, predict=predict)
    monkeypatch.setattr("src.detector.YOLO", lambda _: model)
    monkeypatch.setattr("src.detector.torch.cuda.is_available", lambda: cuda_available)

    detector = PlayerDetector(confidence=0.6)
    assert detector.device == expected_device
    assert detector.detect(frame) == []


def test_missing_input_video_is_rejected(tmp_path: Path) -> None:
    processor = VideoProcessor(detector=SimpleNamespace())
    with pytest.raises(ValueError, match="Input video does not exist"):
        processor.process(tmp_path / "missing.mp4", tmp_path / "output.mp4")


def test_video_output_preserves_size_fps_and_frame_count(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "new-folder" / "output.mp4"
    width, height, fps, frame_count = 64, 48, 12.0, 3

    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), "OpenCV MP4 writing is needed for this test"
    try:
        for _ in range(frame_count):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    finally:
        writer.release()

    detection = {
        "bbox": [5, 20, 40, 45], "confidence": 0.9,
        "class_id": 0, "class_name": "person",
    }
    detector = SimpleNamespace(detect=lambda frame: [detection])
    processed = VideoProcessor(detector).process(input_path, output_path)
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert processed == frame_count
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(fps, abs=0.1)
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == frame_count
        ok, annotated_frame = capture.read()
        assert ok
        assert np.any(annotated_frame != 0), "Boxes and labels should be visible"
    finally:
        capture.release()
