"""Regression tests for decoding failures and output validation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import main
from src.video_processor import VideoProcessor


@pytest.fixture
def video_resources(monkeypatch, tmp_path: Path):
    """Use controlled video metadata without decoding a real video."""
    input_path = tmp_path / "input.mp4"
    input_path.touch()
    metadata = {
        cv2.CAP_PROP_FRAME_WIDTH: 64,
        cv2.CAP_PROP_FRAME_HEIGHT: 48,
        cv2.CAP_PROP_FPS: 12.0,
        cv2.CAP_PROP_FRAME_COUNT: 3,
    }
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = metadata.__getitem__
    capture.read.side_effect = [
        (True, np.zeros((48, 64, 3), dtype=np.uint8)),
        (False, None),
    ]
    writer = Mock()
    writer.isOpened.return_value = True
    writer_factory = Mock(return_value=writer)
    monkeypatch.setattr("src.video_processor.cv2.VideoCapture", lambda _: capture)
    monkeypatch.setattr("src.video_processor.cv2.VideoWriter", writer_factory)
    return input_path, metadata, capture, writer, writer_factory


@pytest.mark.parametrize("property_id", [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT])
def test_odd_resolution_is_rejected_before_writing(video_resources, tmp_path, property_id):
    input_path, metadata, capture, _, writer_factory = video_resources
    metadata[property_id] += 1
    processor = VideoProcessor(SimpleNamespace(detect=lambda frame: []))

    with pytest.raises(ValueError, match="even width and height"):
        processor.process(input_path, tmp_path / "output.mp4")

    writer_factory.assert_not_called()
    capture.release.assert_called_once()


def test_early_decode_failure_is_reported_and_resources_released(video_resources, tmp_path):
    input_path, _, capture, writer, _ = video_resources
    processor = VideoProcessor(SimpleNamespace(detect=lambda frame: []))
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"previous output")

    with pytest.raises(ValueError, match="stopped after 1 of 3"):
        processor.process(input_path, output_path)

    writer.write.assert_called_once()
    writer.release.assert_called_once()
    capture.release.assert_called_once()
    assert output_path.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".output-*.mp4"))


def test_frame_size_mismatch_is_rejected(video_resources, tmp_path):
    input_path, _, capture, writer, _ = video_resources
    capture.read.side_effect = [(True, np.zeros((40, 64, 3), dtype=np.uint8))]
    processor = VideoProcessor(SimpleNamespace(detect=lambda frame: []))

    with pytest.raises(ValueError, match="does not match"):
        processor.process(input_path, tmp_path / "output.mp4")

    writer.write.assert_not_called()
    writer.release.assert_called_once()
    capture.release.assert_called_once()


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fps_is_rejected(video_resources, tmp_path, fps):
    input_path, metadata, capture, _, writer_factory = video_resources
    metadata[cv2.CAP_PROP_FPS] = fps
    processor = VideoProcessor(SimpleNamespace(detect=lambda frame: []))

    with pytest.raises(ValueError, match="invalid FPS"):
        processor.process(input_path, tmp_path / "output.mp4")

    writer_factory.assert_not_called()
    capture.release.assert_called_once()


def test_video_that_cannot_be_opened_is_rejected(video_resources, tmp_path):
    input_path, _, capture, _, writer_factory = video_resources
    capture.isOpened.return_value = False
    processor = VideoProcessor(SimpleNamespace(detect=lambda frame: []))

    with pytest.raises(ValueError, match="Could not open input video"):
        processor.process(input_path, tmp_path / "output.mp4")

    capture.release.assert_called_once()
    writer_factory.assert_not_called()


def test_writer_failure_preserves_existing_output(video_resources, tmp_path):
    input_path, _, capture, writer, _ = video_resources
    writer.isOpened.return_value = False
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"previous output")
    processor = VideoProcessor(SimpleNamespace(detect=lambda frame: []))

    with pytest.raises(RuntimeError, match="Could not create output video"):
        processor.process(input_path, output_path)

    assert output_path.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".output-*.mp4"))
    writer.release.assert_called_once()
    capture.release.assert_called_once()


def test_inference_failure_removes_temporary_output(video_resources, tmp_path):
    input_path, _, capture, writer, _ = video_resources
    detector = Mock()
    detector.detect.side_effect = RuntimeError("Inference failed")
    output_path = tmp_path / "output.mp4"

    with pytest.raises(RuntimeError, match="Inference failed"):
        VideoProcessor(detector).process(input_path, output_path)

    assert not output_path.exists()
    assert not list(tmp_path.glob(".output-*.mp4"))
    writer.release.assert_called_once()
    capture.release.assert_called_once()


def test_success_replaces_output_after_writer_is_closed(video_resources, tmp_path):
    input_path, metadata, _, writer, writer_factory = video_resources
    metadata[cv2.CAP_PROP_FRAME_COUNT] = 1
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"previous output")

    def finish_video():
        assert output_path.read_bytes() == b"previous output"
        temporary_path = Path(writer_factory.call_args.args[0])
        temporary_path.write_bytes(b"finished video")

    writer.release.side_effect = finish_video
    processor = VideoProcessor(SimpleNamespace(detect=lambda frame: []))

    assert processor.process(input_path, output_path) == 1
    assert output_path.read_bytes() == b"finished video"
    assert not list(tmp_path.glob(".output-*.mp4"))
    writer.release.assert_called_once()


def test_invalid_output_path_is_rejected_before_loading_model(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.mp4"
    input_path.touch()
    load_model = Mock()
    monkeypatch.setattr(main, "PlayerDetector", load_model)
    monkeypatch.setattr(main, "parse_args", lambda: SimpleNamespace(
        input=input_path, output=tmp_path / "output.txt",
        confidence=0.5, model="yolo26n.pt",
    ))

    assert main.main() == 1
    assert "must end in .mp4" in capsys.readouterr().err
    load_model.assert_not_called()


def test_output_directory_error_has_no_traceback(video_resources, tmp_path, monkeypatch, capsys):
    input_path, _, capture, _, writer_factory = video_resources
    blocked_directory = tmp_path / "regular-file"
    blocked_directory.touch()
    detector = SimpleNamespace(device="cpu", detect=lambda frame: [])
    monkeypatch.setattr(main, "PlayerDetector", lambda **kwargs: detector)
    monkeypatch.setattr(main, "parse_args", lambda: SimpleNamespace(
        input=input_path, output=blocked_directory / "output.mp4",
        confidence=0.5, model="yolo26n.pt",
    ))

    assert main.main() == 1
    messages = capsys.readouterr()
    assert "Error:" in messages.err
    assert "Traceback" not in messages.err
    assert "Processing complete" not in messages.out
    writer_factory.assert_not_called()
    capture.release.assert_called_once()
