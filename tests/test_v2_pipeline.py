"""V2 video/JSON integration without pretrained weights or a real match."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import main
from src.events import RimROI
from src.video_processor import VideoProcessor


def tiny_video(path, count=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(count):
            writer.write(np.zeros((48, 64, 3), np.uint8))
    finally:
        writer.release()


def components(has_ball=True):
    person = {"track_id": 7, "bbox": [5, 10, 25, 44], "center": (15, 27),
              "confidence": 0.9, "class_id": 0, "class_name": "person"}
    ball = {"bbox": [9, 17, 15, 23], "center": (12, 20),
            "confidence": 0.8, "class_id": 32, "class_name": "sports ball"}
    detector = Mock()
    detector.detect_scene.return_value = (None, [ball] if has_ball else [])
    tracker = Mock(config=SimpleNamespace(track_low_thresh=0.1))
    tracker.track_detections.return_value = [person]
    return detector, tracker


@pytest.mark.parametrize("has_ball,has_roi", [(True, True), (True, False), (False, True), (False, False)])
def test_v2_preserves_video_and_v1_fields_with_or_without_ball_roi(tmp_path, has_ball, has_roi):
    source, output = tmp_path / "input.mp4", tmp_path / "v2.mp4"
    tiny_video(source)
    detector, tracker = components(has_ball)
    processor = VideoProcessor(detector, tracker, enable_basketball=True,
                               rim_roi=RimROI(40, 5, 55, 15) if has_roi else None, show_events=True)
    assert processor.process(source, output) == 4
    assert detector.detect_scene.call_count == tracker.track_detections.call_count == 4
    detector.detect.assert_not_called()
    tracker.track.assert_not_called()
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12)
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 64
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 48
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        assert decoded == 4
    finally:
        capture.release()
    report = json.loads(processor.analytics_output.read_text())
    assert report["video"]["frames_processed"] == 4
    assert report["tracks"][0]["track_id"] == 7
    assert report["units"] == {"distance": "px", "speed": "px/s"}
    assert report["basketball"]["detection_rate"] == int(has_ball)
    assert report["possession_proxy"]["unknown_frames"] == (2 if has_ball else 4)
    assert report["shot_candidates"]["enabled"] == has_roi
    assert report["shot_candidates"]["attempts"] == 0
    assert report["performance"]["processing_seconds"] > 0
    assert report["performance"]["processing_fps"] * report["performance"]["processing_seconds"] == pytest.approx(4)
    assert cv2.imread(str(processor.heatmap_output)).shape == (48, 64, 3)


def test_invalid_roi_fails_before_inference_and_preserves_output(tmp_path):
    source, output = tmp_path / "input.mp4", tmp_path / "v2.mp4"
    tiny_video(source)
    output.write_bytes(b"previous output")
    detector, tracker = components()
    with pytest.raises(ValueError, match="within"):
        VideoProcessor(detector, tracker, enable_basketball=True, rim_roi=RimROI(60, 10, 80, 20)).process(source, output)
    detector.detect_scene.assert_not_called()
    tracker.start.assert_not_called()
    assert output.read_bytes() == b"previous output"


def test_v2_inference_failure_preserves_all_existing_outputs(tmp_path):
    source, output = tmp_path / "input.mp4", tmp_path / "v2.mp4"
    tiny_video(source)
    outputs = [output, tmp_path / "v2_tracking.json", tmp_path / "v2_heatmap.png"]
    for path in outputs:
        path.write_bytes(b"previous output")
    detector, tracker = components()
    detector.detect_scene.side_effect = RuntimeError("Scene inference failed")
    with pytest.raises(RuntimeError, match="Scene inference failed"):
        VideoProcessor(detector, tracker, enable_basketball=True).process(source, output)
    assert all(path.read_bytes() == b"previous output" for path in outputs)
    assert not list(tmp_path.glob(".v2*"))


def test_synthetic_shot_runs_through_ball_proxy_events_and_json(tmp_path):
    """Validate the whole temporal chain; this is not a real shot benchmark."""
    source, output = tmp_path / "input.mp4", tmp_path / "shot.mp4"
    path = [(25, 180)] * 3 + [(50, 160), (75, 120), (90, 90), (95, 65),
                             (100, 20), (100, 15), (100, 35), (100, 55), (100, 70)]
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    assert writer.isOpened()
    try:
        for _ in path:
            writer.write(np.zeros((240, 320, 3), np.uint8))
    finally:
        writer.release()
    detector, tracker = components()
    tracker.track_detections.return_value = [{
        "track_id": 7, "bbox": [10, 150, 40, 230], "center": (25, 190),
        "confidence": 0.9, "class_id": 0, "class_name": "person",
    }]
    predictions = [(None, [{"bbox": [x - 3, y - 3, x + 3, y + 3], "center": (x, y),
                           "confidence": 0.9, "class_id": 32, "class_name": "sports ball"}])
                   for x, y in path]
    detector.detect_scene.side_effect = predictions * 2
    processor = VideoProcessor(detector, tracker, enable_basketball=True,
                               rim_roi=RimROI(90, 30, 110, 50), show_events=True)
    # Reusing the processor must begin a fresh event/ball/possession lifecycle.
    for _ in range(2):
        assert processor.process(source, output) == len(path)
        report = json.loads(processor.analytics_output.read_text())
        assert report["basketball"]["frames_detected"] == len(path)
        assert report["basketball"]["primary_ball_tracks"] == 1
        assert report["shot_candidates"]["attempts"] == 1
        assert report["shot_candidates"]["made_candidates"] == 1
        assert report["shot_candidates"]["unknown_outcomes"] == 0
        shot = report["shot_candidates"]["shots"][0]
        assert shot["shot_id"] == shot["ball_track_id"] == 1
        assert shot["player_track_id"] == 7
        assert report["events"][-1]["type"] == "made_shot_candidate"
        assert report["events"][-1]["timestamp_seconds"] == pytest.approx((len(path) - 1) / 30)


@pytest.mark.parametrize("flag,value", [("--ball-confidence", "nan"), ("--ball-confidence", "2"),
                                      ("--possession-min-frames", "0"), ("--possession-distance-threshold", "0")])
def test_invalid_v2_cli_configuration(monkeypatch, flag, value):
    monkeypatch.setattr("sys.argv", ["main.py", "--input", "in.mp4", "--output", "out.mp4", flag, value])
    with pytest.raises(SystemExit) as error:
        main.parse_args()
    assert error.value.code == 2
