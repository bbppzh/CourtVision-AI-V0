"""Exercise result extraction and real ByteTrack association without YOLO weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from src.tracker import PlayerTracker, load_tracker_config


FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def result(rows, tracked=True):
    columns = 7 if tracked else 6
    return Results(FRAME.copy(), "", {0: "person", 1: "bicycle"}, boxes=np.array(rows, dtype=np.float32).reshape(-1, columns))


def test_extracts_id_center_and_only_visible_people():
    predictions = result([
        [2.5, 4.5, 12.5, 24.5, 7, 0.9, 0],
        [2, 4, 12, 24, 8, 0.9, 1],
        [2, 4, 12, 24, 9, 0.2, 0],
        [2, 4, 12, 24, -1, 0.9, 0],
        [2, 4, 12, 24, 1.5, 0.9, 0],
        [2, 4, 12, 24, float("nan"), 0.9, 0],
        [12, 4, 2, 24, 10, 0.9, 0],
    ])
    people = PlayerTracker.extract_tracked_people(predictions, 0.5)
    assert len(people) == 1
    assert people[0]["track_id"] == 7
    assert people[0]["bbox"] == [2, 4, 12, 24]
    assert people[0]["center"] == (7.5, 14.5)
    assert people[0]["class_name"] == "person"
    assert people[0]["confidence"] == pytest.approx(0.9)


def test_missing_ids_and_empty_results_are_safe():
    assert PlayerTracker.extract_tracked_people(result([[2, 4, 12, 24, 0.9, 0]], tracked=False), 0.5) == []
    assert PlayerTracker.extract_tracked_people(result([]), 0.5) == []
    assert PlayerTracker.extract_tracked_people(SimpleNamespace(boxes=None), 0.5) == []


def test_real_bytetrack_retains_id_through_weak_detection_and_resets():
    def predict(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert kwargs["classes"] == [0]
        assert kwargs["conf"] == 0.1
        assert kwargs["device"] == "cpu"
        return [next(predictions)]

    predictions = iter([
        result([[10, 10, 25, 40, 0.9, 0]], tracked=False),
        result([[11, 10, 26, 40, 0.2, 0]], tracked=False),
        result([[12, 10, 27, 40, 0.9, 0]], tracked=False),
        result([[10, 10, 25, 40, 0.9, 0]], tracked=False),
    ])
    detector = SimpleNamespace(confidence=0.5, device="cpu", model=SimpleNamespace(predict=predict, names={0: "person"}))
    tracker = PlayerTracker(detector)
    tracker.start()
    first_id = tracker.track(FRAME)[0]["track_id"]
    assert tracker.track(FRAME) == []  # Weak detection associates but is hidden.
    assert tracker.track(FRAME)[0]["track_id"] == first_id
    tracker.start()
    assert tracker._tracker.frame_id == 0
    assert tracker._tracker.lost_stracks == []
    assert tracker.track(FRAME)[0]["track_id"] == 1  # IDs are local to each video.


def test_empty_frame_advances_tracker_lifecycle():
    detector = SimpleNamespace(confidence=0.5, device="cpu", model=Mock(names={0: "person"}))
    detector.model.predict.return_value = [result([], tracked=False)]
    tracker = PlayerTracker(detector)
    with pytest.raises(RuntimeError, match="Start the tracker"):
        tracker.track(FRAME)
    tracker.start()
    assert tracker.track(FRAME) == []
    assert tracker._tracker.frame_id == 1


@pytest.mark.parametrize("change", [
    {"tracker_type": "botsort"}, {"track_buffer": 0},
    {"track_low_thresh": 0.9}, {"match_thresh": float("nan")},
    {"fuse_score": "yes"},
])
def test_invalid_tracker_settings_are_rejected(tmp_path, change):
    from ultralytics.utils import YAML

    settings = vars(load_tracker_config("bytetrack.yaml")).copy()
    settings.update(change)
    path = tmp_path / "tracker.yaml"
    YAML.save(path, settings)
    with pytest.raises(ValueError, match="ByteTrack|bytetrack"):
        PlayerTracker(Mock(), str(path))


def test_missing_tracker_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="existing YAML"):
        load_tracker_config(str(tmp_path / "missing.yaml"))
