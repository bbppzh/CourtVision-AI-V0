"""Verify V1 sidecars and metadata with three tiny frames and a fake tracker."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.analytics import MotionAnalytics
from src.utils import validate_tracking_paths
from src.video_processor import VideoProcessor


def small_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize("has_people", [True, False])
def test_tracking_roundtrip_and_exports(tmp_path, has_people):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "nested" / "tracked.mp4"
    small_video(input_path)
    tracker = Mock()
    tracker.track.side_effect = [
        [{"track_id": 7, "center": (10 + 3 * i, 20 + 4 * i), "bbox": [5, 5, 25, 40], "confidence": 0.9, "class_id": 0, "class_name": "person"}] if has_people else []
        for i in range(3)
    ]
    processor = VideoProcessor(SimpleNamespace(), tracker, show_speed=True)
    assert processor.process(input_path, output_path) == 3
    tracker.start.assert_called_once()
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0)
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 64
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 48
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        assert decoded == 3
    finally:
        capture.release()
    report = json.loads(processor.analytics_output.read_text())
    assert report["video"]["frames_processed"] == 3
    assert len(report["tracks"]) == int(has_people)
    assert processor.unique_tracks == int(has_people)
    if has_people:
        track = report["tracks"][0]
        assert track["frames_seen"] == 3
        assert track["cumulative_pixel_distance"] == pytest.approx(10)
        assert track["average_pixel_speed"] == pytest.approx(60)
    image = cv2.imread(str(processor.heatmap_output))
    assert image.shape == (48, 64, 3)
    assert bool(image.any()) == has_people


def test_export_failure_preserves_all_existing_outputs(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    small_video(input_path)
    output = tmp_path / "tracked.mp4"
    analytics = tmp_path / "tracked_tracking.json"
    heatmap = tmp_path / "tracked_heatmap.png"
    for path in (output, analytics, heatmap):
        path.write_bytes(b"previous output")
    tracker = Mock()
    tracker.track.return_value = []
    monkeypatch.setattr(MotionAnalytics, "save_json", Mock(side_effect=OSError("Report export failed")))
    with pytest.raises(OSError, match="Report export failed"):
        VideoProcessor(SimpleNamespace(), tracker).process(input_path, output)
    for path in (output, analytics, heatmap):
        assert path.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".tracked*"))


def test_paths_cannot_overwrite_input_or_share_sidecar(tmp_path):
    input_path = tmp_path / "input.png"
    input_path.touch()
    with pytest.raises(ValueError, match="must all be different"):
        validate_tracking_paths(input_path, tmp_path / "out.mp4", tmp_path / "out.json", input_path)


def test_invalid_trajectory_length_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        VideoProcessor(SimpleNamespace(), trajectory_length=0)


def test_final_rename_failure_restores_previous_outputs(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    small_video(input_path)
    outputs = [tmp_path / name for name in ("tracked.mp4", "tracked_tracking.json", "tracked_heatmap.png")]
    for path in outputs:
        path.write_bytes(b"previous output")
    original_replace = Path.replace
    failed_once = False

    def fail_last_publish(path, destination):
        nonlocal failed_once
        if Path(destination) == outputs[2] and not failed_once:
            failed_once = True
            raise OSError("Simulated final rename failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_last_publish)
    tracker = Mock()
    tracker.track.return_value = []
    with pytest.raises(OSError, match="final rename failure"):
        VideoProcessor(SimpleNamespace(), tracker).process(input_path, outputs[0])
    for path in outputs:
        assert path.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".tracked*"))


def test_reusing_processor_resets_state_and_derives_new_sidecar_names(tmp_path):
    input_path = tmp_path / "input.mp4"
    small_video(input_path)
    tracker = Mock()
    tracker.track.return_value = []
    processor = VideoProcessor(SimpleNamespace(), tracker)
    processor.process(input_path, tmp_path / "first.mp4")
    first_analytics = processor.analytics
    first_report = (tmp_path / "first_tracking.json").read_bytes()
    processor.process(input_path, tmp_path / "second.mp4")
    assert processor.analytics is not first_analytics
    assert processor.analytics_output.name == "second_tracking.json"
    assert processor.heatmap_output.name == "second_heatmap.png"
    assert (tmp_path / "first_tracking.json").read_bytes() == first_report
    assert tracker.start.call_count == 2
