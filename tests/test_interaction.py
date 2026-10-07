"""Temporal, normalized proximity is a possession proxy, not ground truth."""

import pytest

from src.interaction import PossessionEstimator, normalized_box_distance


def player(identity, bbox):
    return {"track_id": identity, "bbox": bbox}


def ball(x, y=40, identity=1):
    return {"center": (x, y), "ball_track_id": identity, "confidence": 0.8}


def test_box_distance_is_zero_inside_and_normalized_outside():
    assert normalized_box_distance((15, 20), [10, 10, 30, 70]) == 0
    assert normalized_box_distance((50, 40), [10, 10, 30, 70]) == pytest.approx(20 / (4000 ** 0.5))


def test_confirmation_and_challenger_hysteresis():
    estimator = PossessionEstimator(min_frames=3)
    players = [player(7, [10, 10, 30, 70]), player(8, [55, 10, 75, 70])]
    assert estimator.update(ball(20), players, 0) is None
    assert estimator.update(ball(20), players, 1) is None
    assert estimator.update(ball(20), players, 2)["player_track_id"] == 7
    assert estimator.update(ball(50), players, 3)["player_track_id"] == 7
    assert estimator.update(ball(50), players, 4)["player_track_id"] == 7
    assert estimator.update(ball(50), players, 5)["player_track_id"] == 8


def test_distant_ball_and_overlapping_players_are_unknown():
    estimator = PossessionEstimator(min_frames=1)
    person = player(7, [10, 10, 30, 70])
    assert estimator.update(ball(250, 200), [person], 0) is None
    assert estimator.update(ball(20), [person, player(8, person["bbox"])], 1) is None


@pytest.mark.parametrize("missing_ball,missing_player", [(True, False), (False, True)])
def test_missing_inputs_reset_confirmation(missing_ball, missing_player):
    estimator = PossessionEstimator(min_frames=2)
    players = [player(7, [10, 10, 30, 70])]
    estimator.update(ball(20), players, 0)
    assert estimator.update(ball(20), players, 1) is not None
    assert estimator.update(None if missing_ball else ball(20), [] if missing_player else players, 2) is None
    assert estimator.update(ball(20), players, 3) is None


def test_frame_gap_and_new_ball_id_do_not_reuse_confirmation():
    estimator = PossessionEstimator(min_frames=2)
    players = [player(7, [10, 10, 30, 70])]
    estimator.update(ball(20), players, 0)
    assert estimator.update(ball(20), players, 1) is not None
    assert estimator.update(ball(20), players, 3) is None
    assert estimator.update(ball(20, identity=2), players, 4) is None


@pytest.mark.parametrize("settings", [{"distance_threshold": 0}, {"distance_threshold": float("nan")}, {"min_frames": 0}])
def test_invalid_possession_settings(settings):
    with pytest.raises(ValueError):
        PossessionEstimator(**settings)
