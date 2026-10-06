"""Sports-ball filtering and association checks without model downloads."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from src.ball_tracker import BallTracker
from src.detector import PlayerDetector
from src.tracker import PlayerTracker


def ball(x, y, confidence=0.8):
    return {"bbox": [int(x - 3), int(y - 3), int(x + 3), int(y + 3)],
            "center": (x, y), "confidence": confidence, "class_id": 32, "class_name": "sports ball"}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scene_uses_one_inference_and_resolves_model_class_names(monkeypatch, device):
    frame = np.zeros((48, 64, 3), np.uint8)
    names = {0: "person", 5: "sports ball", 9: "car"}
    result = Results(frame, "", names, boxes=np.array([
        [1, 2, 20, 40, 0.2, 0], [25, 15, 31, 21, 0.8, 5],
        [30, 15, 36, 21, 0.1, 5], [40, 15, 46, 21, 0.9, 9],
    ], dtype=np.float32))
    def predict(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert kwargs["classes"] == [0, 5]
        assert kwargs["conf"] == 0.1
        assert kwargs["device"] == device
        return [result]
    model = SimpleNamespace(task="detect", names=names, predict=Mock(side_effect=predict))
    monkeypatch.setattr("src.detector.YOLO", lambda _: model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    detector = PlayerDetector()
    people, balls = detector.detect_scene(frame, 0.1, 0.25)
    model.predict.assert_called_once()
    assert len(people) == 1 and float(people.conf[0]) == pytest.approx(0.2)
    assert len(balls) == 1 and balls[0]["class_id"] == 5
    assert balls[0]["center"] == (28, 18)
    assert balls[0]["class_name"] == "sports ball"


def test_missing_sports_ball_class_is_a_clear_v2_error(monkeypatch):
    monkeypatch.setattr("src.detector.YOLO", lambda _: SimpleNamespace(task="detect", names={0: "person"}))
    detector = PlayerDetector()
    with pytest.raises(ValueError, match="sports ball"):
        detector.detect_scene(np.zeros((48, 64, 3), np.uint8), 0.1)
    assert detector.filter_sports_ball_detections(SimpleNamespace(boxes=None), 0.25, 32) == []


def test_shared_predictions_feed_real_bytetrack_without_ball_ids(monkeypatch):
    frame = np.zeros((48, 64, 3), np.uint8)
    names = {0: "person", 32: "sports ball"}
    result = Results(frame, "", names, boxes=np.array([
        [5, 5, 25, 44, 0.9, 0], [30, 15, 36, 21, 0.8, 32],
    ], dtype=np.float32))
    model = SimpleNamespace(task="detect", names=names, predict=Mock(return_value=[result]))
    monkeypatch.setattr("src.detector.YOLO", lambda _: model)
    detector = PlayerDetector()
    tracker = PlayerTracker(detector)
    tracker.start()
    for _ in range(3):
        people, balls = detector.detect_scene(frame, tracker.config.track_low_thresh)
        tracks = tracker.track_detections(frame, people)
        assert [track["track_id"] for track in tracks] == [1]
        assert all(track["class_name"] == "person" for track in tracks)
        assert len(balls) == 1
    assert model.predict.call_count == 3


def test_continuity_wins_over_a_distant_high_confidence_ball():
    tracker = BallTracker(320, 240)
    first = tracker.update([ball(20, 20, 0.7), ball(250, 200, 0.6)], 0)
    second = tracker.update([ball(250, 200, 0.99), ball(25, 20, 0.4)], 1)
    assert first["ball_track_id"] == second["ball_track_id"] == 1
    assert second["center"] == (25, 20)


def test_initial_ties_are_deterministic_and_history_is_bounded():
    tracker = BallTracker(320, 240, history_length=3)
    assert tracker.update([ball(30, 20), ball(20, 20)], 0)["center"] == (20, 20)
    for frame in range(1, 8):
        tracker.update([ball(20 + frame, 20)], frame)
    assert len(tracker.history) == 3
    assert tracker.history[0][0] == 5


def test_missing_observation_breaks_trail_and_long_gap_changes_ball_id():
    tracker = BallTracker(320, 240, max_gap_frames=2)
    tracker.update([ball(20, 20)], 0)
    assert tracker.update([], 1) is None
    assert tracker.update([ball(22, 20)], 2)["ball_track_id"] == 1
    assert list(tracker.history) == [(2, 22, 20)]
    assert tracker.update([ball(250, 200)], 3) is None
    assert tracker.update([], 5) is None
    assert not tracker.history
    assert tracker.update([ball(250, 200)], 6)["ball_track_id"] == 2
    tracker.reset()
    assert tracker.update([ball(10, 10)], 0)["ball_track_id"] == 1


def test_extreme_scene_boxes_are_rejected():
    tracker = BallTracker(320, 240)
    candidate = ball(20, 20)
    candidate["bbox"] = [0, 0, 300, 200]
    assert tracker.update([candidate, ball(-10, 20)], 0) is None


FPS_CASES = [24, 30, 60, 120]
# A fast diagonal flight through the middle of a 320x240 frame, in px/s.
FLIGHT_SPEED = 600.0
FLIGHT_ORIGIN = (60.0, 200.0)
FLIGHT_VELOCITY = (60.0, -240.0)
FLIGHT_SECONDS = 0.8


def moving_ball(origin, velocity, fps, step):
    return ball(origin[0] + velocity[0] * step / fps, origin[1] + velocity[1] * step / fps)


@pytest.mark.parametrize("fps", FPS_CASES)
def test_one_physical_speed_is_tracked_at_every_frame_rate(fps):
    """The same flight must keep one ball ID at 24, 30, 60 and 120 FPS.

    The old pixels-per-frame gate let a constant 600 px/s ball through at
    24 FPS but rejected it at 120 FPS, where the same speed moves 12x fewer
    pixels between frames relative to the gate.
    """
    tracker = BallTracker(320, 240, fps=float(fps))
    ids = set()
    for step in range(int(FLIGHT_SECONDS * fps)):
        track = tracker.update([moving_ball(FLIGHT_ORIGIN, FLIGHT_VELOCITY, fps, step)], step)
        assert track is not None, f"{fps} FPS lost the ball at step {step}"
        ids.add(track["ball_track_id"])
    assert ids == {1}


@pytest.mark.parametrize("fps", FPS_CASES)
def test_sustained_change_of_speed_keeps_one_ball_id(fps):
    """A flight that slows near the apex must not be dropped or re-numbered.

    A fixed pixels-per-frame gate treats the slowdown as a different ball
    because the prediction keeps extrapolating the fast approach.
    """
    tracker = BallTracker(320, 240, fps=float(fps))
    ids, y = set(), 200.0
    for step in range(int(1.4 * fps)):
        y -= (600.0 if y > 120 else 60.0) / fps
        track = tracker.update([ball(100.0, y)], step)
        assert track is not None, f"{fps} FPS dropped the ball at step {step} (y={y:.1f})"
        ids.add(track["ball_track_id"])
    assert ids == {1}


def test_gap_keeps_the_id_but_never_fabricates_history():
    tracker = BallTracker(320, 240, max_gap_frames=5, fps=30.0)
    tracker.update([ball(100, 30)], 0)
    assert tracker.update([], 1) is None
    assert tracker.update([], 2) is None
    assert not tracker.history
    resumed = tracker.update([ball(104, 40)], 3)
    assert resumed is not None and resumed["ball_track_id"] == 1
    # History restarts at the resumed observation; the gap is never bridged.
    assert list(tracker.history) == [(3, 104, 40)]


def test_gap_beyond_the_limit_starts_a_new_ball_id():
    tracker = BallTracker(320, 240, max_gap_frames=5, fps=30.0)
    tracker.update([ball(100, 30)], 0)
    assert tracker.update([], 3) is None
    restarted = tracker.update([ball(100, 30)], 6)
    assert restarted is not None and restarted["ball_track_id"] == 2


def test_ball_tracker_rejects_invalid_fps():
    with pytest.raises(ValueError, match="FPS"):
        BallTracker(320, 240, fps=0)
    with pytest.raises(ValueError, match="FPS"):
        BallTracker(320, 240, fps=float("nan"))
