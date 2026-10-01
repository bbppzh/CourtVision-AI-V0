"""Small heatmap and drawing checks independent of YOLO weights."""

import cv2
import numpy as np
import pytest

from src.heatmap import MovementHeatmap
from src.utils import draw_tracked_person, track_color


def test_heatmap_counts_centers_and_ignores_outside_points():
    heatmap = MovementHeatmap(64, 48)
    heatmap.update([{"center": (10.5, 20.5)}, {"center": (63.9, 47.9)}, {"center": (-1, 5)}, {"center": (64, 5)}, {"center": (float("nan"), 5)}])
    heatmap.update([{"center": (10.5, 20.5)}])
    assert heatmap.activity[20, 10] == 2
    assert heatmap.activity[47, 63] == 1
    assert heatmap.activity.sum() == 3
    image = heatmap.render()
    assert image.shape == (48, 64, 3)
    assert image.dtype == np.uint8
    assert image.any()


def test_no_people_produces_valid_black_png(tmp_path):
    heatmap = MovementHeatmap(64, 48)
    output = tmp_path / "nested" / "heatmap.png"
    heatmap.save(output)
    image = cv2.imread(str(output))
    assert image.shape == (48, 64, 3)
    assert not image.any()


def test_heatmap_write_failure_preserves_existing_file(tmp_path, monkeypatch):
    output = tmp_path / "heatmap.png"
    output.write_bytes(b"previous image")
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(RuntimeError, match="Could not write heatmap"):
        MovementHeatmap(64, 48).save(output)
    assert output.read_bytes() == b"previous image"
    assert not list(tmp_path.glob(".heatmap-*.png"))


def test_trajectory_draws_outside_box_and_single_point_is_safe():
    person = {"track_id": 7, "bbox": [40, 5, 60, 25], "confidence": 0.9}
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    draw_tracked_person(frame, person, [(5, 35), (25, 35)])
    assert frame[35, 15].any()
    draw_tracked_person(frame, person, [(5, 35)], speed=123.0)
    assert track_color(7) == track_color(7)
    assert track_color(7) != track_color(8)
