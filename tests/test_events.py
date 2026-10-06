"""Synthetic temporal evidence for conservative shot outcomes."""

import pytest

from src.events import EventAnalyzer, RimROI


PLAYER = {"track_id": 7, "bbox": [10, 150, 40, 230]}
PROXY = {"player_track_id": 7, "heuristic_score": 0.8}
ROI = RimROI(90, 30, 110, 50)
MADE_PATH = [(25, 180), (25, 180), (25, 180), (85, 115), (90, 90),
             (95, 65), (100, 20), (100, 15), (100, 35), (100, 55), (100, 70)]


def ball(center, identity=1):
    return {"center": center, "ball_track_id": identity, "confidence": 0.9}


def feed(engine, path):
    for frame, center in enumerate(path):
        engine.update(None if center is None else ball(center), PROXY if frame < 3 else None, [PLAYER], frame)


def test_valid_roi_parsing_and_dimensions():
    roi = RimROI.parse("90, 30,110,50")
    assert roi == ROI
    roi.validate_dimensions(320, 240)
    with pytest.raises(ValueError, match="within"):
        roi.validate_dimensions(100, 40)


@pytest.mark.parametrize("value", ["1,2,3", "a,2,3,4", "1.5,2,3,4", "3,2,1,4", "1,4,3,2", "-1,2,3,4"])
def test_invalid_rim_roi(value):
    with pytest.raises(ValueError):
        RimROI.parse(value)


def test_upward_motion_alone_does_not_trigger_shot():
    engine = EventAnalyzer(30, ROI)
    for frame, point in enumerate(MADE_PATH[3:]):
        engine.update(ball(point), None, [], frame)
    assert engine.shots == []


def test_release_ascending_approach_and_crossing_make_candidate():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH)
    assert len(engine.shots) == 1
    shot = engine.shots[0]
    assert shot["start_frame"] == 3 and shot["decision_frame"] == 5
    assert shot["player_track_id"] == 7
    assert shot["outcome"] == "made_candidate"
    events = [event for event in engine.events if event["shot_id"] is not None]
    assert [event["type"] for event in events] == ["shot_attempt_candidate", "made_shot_candidate"]
    assert events[0]["timestamp_seconds"] == pytest.approx(5 / 30)
    assert engine.overlay(11) == "MADE CANDIDATE"
    assert engine.overlay(40) is None


def test_one_inside_point_and_disappearance_remain_unknown():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH[:8] + [(100, 40)] + [None] * 6)
    assert engine.summary()["unknown_outcomes"] == 1
    assert engine.summary()["made_candidates"] == 0


def test_gap_cannot_invent_rim_crossing():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH[:8] + [None, (100, 55), (100, 70), (100, 85)])
    engine.finalize(11)
    assert engine.shots[0]["outcome"] == "unknown"


def test_clear_outside_downward_sequence_can_be_missed_candidate():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH[:8] + [(140, 35), (145, 55), (150, 70)])
    assert engine.shots[0]["outcome"] == "missed_candidate"


def test_incomplete_candidate_is_unknown_at_end():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH[:6])
    engine.finalize(5)
    assert engine.shots[0]["outcome"] == "unknown"
    assert engine.events[-1]["type"] == "shot_outcome_unknown"


def test_old_holder_evidence_keeps_uncertain_attribution_null():
    engine = EventAnalyzer(30, ROI)
    # Still near the old box, but no fresh stable proxy is available for 0.5 s.
    path = MADE_PATH[:3] + [(25, 180)] * 15 + MADE_PATH[3:]
    feed(engine, path)
    assert len(engine.shots) == 1
    assert engine.shots[0]["player_track_id"] is None


def test_ball_identity_change_finishes_candidate_as_unknown():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH[:6])
    engine.update(ball((100, 20), identity=2), None, [], 6)
    assert engine.shots[0]["outcome"] == "unknown"


def test_duplicates_need_new_possession_and_release():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH)
    for frame in range(11, 80):
        engine.update(ball((100, 20 if frame % 2 else 60)), None, [], frame)
    assert len(engine.shots) == 1
    assert sum(event["type"] == "shot_attempt_candidate" for event in engine.events) == 1


def test_no_roi_keeps_proxy_timeline_but_disables_shots():
    engine = EventAnalyzer(30)
    feed(engine, MADE_PATH)
    assert not engine.summary()["enabled"]
    assert engine.shots == []
    assert engine.events[0]["type"] == "possession_proxy_change"
