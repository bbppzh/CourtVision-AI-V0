"""Pixel motion and atomic reporting tests without model inference."""

import json
from pathlib import Path

import pytest

from src.analytics import MotionAnalytics


def track(track_id: int, center: tuple[float, float]) -> dict:
    return {
        "track_id": track_id, "center": center, "bbox": [0, 0, 10, 10],
        "confidence": 0.9, "class_id": 0, "class_name": "person",
    }


def test_three_four_five_displacement_and_speed() -> None:
    analytics = MotionAnalytics(width=3, height=4, fps=10.0)
    analytics.update([track(1, (0.0, 0.0))], 0)
    analytics.update([track(1, (3.0, 4.0))], 1)

    summary = analytics.to_dict(2)["tracks"][0]
    assert analytics.unique_tracks == 1
    assert analytics.speed(1) == pytest.approx(50.0)
    assert summary["frames_seen"] == 2
    assert summary["cumulative_pixel_distance"] == pytest.approx(5.0)
    assert summary["average_pixel_speed"] == pytest.approx(50.0)
    assert summary["max_pixel_speed"] == pytest.approx(50.0)
    assert summary["normalized_distance"] == pytest.approx(1.0)


def test_speed_window_and_maximum_use_smoothed_speed() -> None:
    analytics = MotionAnalytics(100, 100, fps=2.0, smoothing_window=2)
    for frame_index, x in enumerate([0.0, 1.0, 6.0, 7.0]):
        analytics.update([track(1, (x, 0.0))], frame_index)

    summary = analytics.to_dict(4)["tracks"][0]
    assert analytics.speed(1) == pytest.approx(6.0)
    assert summary["max_pixel_speed"] == pytest.approx(6.0)
    assert summary["average_pixel_speed"] == pytest.approx(7.0 / 1.5)
    assert len(analytics._tracks[1].speed_samples) == 2


def test_zero_displacement_and_one_observation_are_safe() -> None:
    analytics = MotionAnalytics(64, 48, fps=12.0)
    analytics.update([track(1, (2.0, 3.0)), track(2, (9.0, 9.0))], 0)
    analytics.update([track(1, (2.0, 3.0))], 1)

    assert analytics.speed(1) == 0.0
    assert analytics.speed(2) == 0.0
    assert analytics.speed(99) == 0.0
    for summary in analytics.to_dict(2)["tracks"]:
        assert summary["cumulative_pixel_distance"] == 0.0
        assert summary["average_pixel_speed"] == 0.0
        assert summary["max_pixel_speed"] == 0.0


def test_missing_frames_do_not_add_distance_or_time() -> None:
    analytics = MotionAnalytics(100, 100, fps=10.0)
    analytics.update([track(1, (0.0, 0.0))], 0)
    analytics.update([track(1, (3.0, 4.0))], 1)
    analytics.update([], 2)
    assert analytics.speed(1) == 0.0
    analytics.update([track(1, (100.0, 100.0))], 3)
    assert analytics.speed(1) == 0.0
    analytics.update([track(1, (100.0, 102.0))], 4)

    summary = analytics.to_dict(5)["tracks"][0]
    assert analytics.speed(1) == pytest.approx(20.0)
    assert summary["frames_seen"] == 4
    assert summary["cumulative_pixel_distance"] == pytest.approx(7.0)
    assert summary["average_pixel_speed"] == pytest.approx(35.0)
    assert summary["max_pixel_speed"] == pytest.approx(50.0)


def test_skipped_frame_index_resets_speed_window() -> None:
    analytics = MotionAnalytics(100, 100, fps=10.0)
    analytics.update([track(1, (0.0, 0.0))], 0)
    analytics.update([track(1, (3.0, 4.0))], 1)
    analytics.update([track(1, (100.0, 100.0))], 4)

    assert analytics.speed(1) == 0.0
    assert analytics.to_dict(5)["tracks"][0]["cumulative_pixel_distance"] == 5.0


@pytest.mark.parametrize("parameter", ["width", "height", "smoothing_window"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_positive_integer_configuration(parameter: str, value) -> None:
    configuration = {"width": 64, "height": 48, "fps": 12.0, "smoothing_window": 5}
    configuration[parameter] = value
    with pytest.raises(ValueError, match="positive integer"):
        MotionAnalytics(**configuration)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fps_is_rejected(fps: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MotionAnalytics(64, 48, fps)


def test_saved_report_is_valid_sorted_json(tmp_path: Path) -> None:
    analytics = MotionAnalytics(64, 48, fps=12.0)
    analytics.update([track(9, (2.0, 3.0)), track(1, (1.0, 1.0))], 0)
    destination = tmp_path / "nested" / "motion.json"
    analytics.save_json(destination, frames_processed=1)

    report = json.loads(destination.read_text())
    assert report["video"] == {"width": 64, "height": 48, "fps": 12.0, "frames_processed": 1}
    assert report["coordinate_system"] == "image bounding_box_center"
    assert report["units"] == {"distance": "px", "speed": "px/s"}
    assert "perspective" in report["scientific_warning"]
    assert "calibration" in report["scientific_warning"]
    assert [item["track_id"] for item in report["tracks"]] == [1, 9]
    assert not list(destination.parent.glob(".motion.json-*.tmp"))


def test_nonfinite_json_preserves_prior_file_and_removes_temp(tmp_path: Path) -> None:
    analytics = MotionAnalytics(64, 48, fps=12.0)
    analytics.update([track(1, (0.0, 0.0))], 0)
    analytics.update([track(1, (float("nan"), 0.0))], 1)
    destination = tmp_path / "motion.json"
    destination.write_text("prior report")

    with pytest.raises(ValueError, match="JSON compliant"):
        analytics.save_json(destination, frames_processed=2)

    assert destination.read_text() == "prior report"
    assert not list(tmp_path.glob(".motion.json-*.tmp"))


def test_failed_replace_preserves_prior_file_and_removes_temp(tmp_path: Path, monkeypatch) -> None:
    analytics = MotionAnalytics(64, 48, fps=12.0)
    destination = tmp_path / "motion.json"
    destination.write_text("prior report")

    def fail_replace(source: Path, target: Path) -> None:
        assert target == destination
        raise OSError("Replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Replace failed"):
        analytics.save_json(destination, frames_processed=0)

    assert destination.read_text() == "prior report"
    assert not list(tmp_path.glob(".motion.json-*.tmp"))
