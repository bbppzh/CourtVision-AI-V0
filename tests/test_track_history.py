"""Short trail tests using plain track records and no model weights."""

import pytest

from src.track_history import TrackHistory


def track(track_id: int, center: tuple[float, float]) -> dict:
    return {
        "track_id": track_id, "center": center, "bbox": [0, 0, 10, 10],
        "confidence": 0.9, "class_id": 0, "class_name": "person",
    }


def test_history_retains_only_the_latest_points() -> None:
    history = TrackHistory(max_length=2)
    for frame_index in range(3):
        history.update([track(1, (float(frame_index), 0.0))], frame_index)

    assert history.get_points(1) == [(1.0, 0.0), (2.0, 0.0)]
    assert history.histories[1].maxlen == 2
    points = history.get_points(1)
    points.clear()
    assert len(history.get_points(1)) == 2
    assert history.get_points(99) == []


def test_missing_frame_starts_a_new_trail() -> None:
    history = TrackHistory()
    history.update([track(1, (0.0, 0.0))], 0)
    history.update([track(1, (3.0, 4.0))], 1)
    history.update([], 2)
    history.update([track(1, (100.0, 100.0))], 3)

    assert history.get_points(1) == [(100.0, 100.0)]


def test_skipped_frame_index_starts_a_new_trail() -> None:
    history = TrackHistory()
    history.update([track(1, (0.0, 0.0))], 0)
    history.update([track(1, (3.0, 4.0))], 2)

    assert history.get_points(1) == [(3.0, 4.0)]


def test_empty_frames_remove_stale_histories() -> None:
    history = TrackHistory(max_stale_frames=2)
    history.update([track(1, (0.0, 0.0))], 0)
    history.update([], 2)
    assert history.get_points(1) == [(0.0, 0.0)]

    history.update([], 3)
    assert history.histories == {}
    assert history._last_seen == {}


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
@pytest.mark.parametrize("parameter", ["max_length", "max_stale_frames"])
def test_invalid_history_configuration(parameter: str, value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TrackHistory(**{parameter: value})
