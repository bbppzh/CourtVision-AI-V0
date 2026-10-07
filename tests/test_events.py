"""Synthetic temporal evidence for conservative shot outcomes."""

import pytest

from src.events import EventAnalyzer, RimROI


PLAYER = {"track_id": 7, "bbox": [10, 150, 40, 230]}
PROXY = {"player_track_id": 7, "heuristic_score": 0.8}
ROI = RimROI(90, 30, 110, 50)
# The ball rises above the rim, then leaves through the *bottom* of the rim
# opening. Leaving through the side instead (drifting right while still inside
# the band) does not complete a passage, so it is not a made candidate.
MADE_PATH = [(25, 180), (25, 180), (25, 180), (85, 115), (90, 90),
             (95, 65), (100, 20), (100, 15), (100, 35), (100, 55), (100, 70)]


def ball(center, identity=1):
    return {"center": center, "ball_track_id": identity, "confidence": 0.9}


def feed(engine, path):
    for frame, center in enumerate(path):
        engine.update(None if center is None else ball(center), PROXY if frame < 3 else None, [PLAYER], frame)


def build_samples(waypoints, factor):
    """Turn (x, y, canonical-duration) waypoints into a continuous trajectory.

    Scaling every duration by the same factor is a consistent time warp: the
    geometry is unchanged and pixel velocities fall by that factor, so the same
    flight can be sampled at any frame rate.
    """
    points = [(x, y) for x, y, _ in waypoints]
    spans = [duration * factor for _, _, duration in waypoints]
    samples = [(points[0][0], points[0][1], 0.0)]
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        span = spans[index - 1]
        steps = max(2, int(span / (1 / 480)))
        for step in range(1, steps + 1):
            fraction = step / steps
            samples.append((
                x0 + (x1 - x0) * fraction,
                y0 + (y1 - y0) * fraction,
                samples[-1][2] + span / steps,
            ))
    return samples


def at(samples, time):
    """Linear lookup of the trajectory position at an absolute time."""
    if time >= samples[-1][2]:
        return samples[-1][0], samples[-1][1]
    low, high = 0, len(samples) - 1
    while low < high:
        middle = (low + high) // 2
        if samples[middle][2] < time:
            low = middle + 1
        else:
            high = middle
    return samples[low][0], samples[low][1]


def analyze_path(samples, fps, duration, missing=()):
    """Feed one continuous path, sampled at `fps`, through the analyzer."""
    engine = EventAnalyzer(float(fps), ROI)
    steps = int(duration * fps)
    for step in range(steps):
        if step in missing:
            engine.update(None, None, [PLAYER], step)
            continue
        x, y = at(samples, step / fps)
        engine.update(ball((x, y)), PROXY if step < 3 else None, [PLAYER], step)
    engine.finalize(steps - 1)
    return engine


# The reported sequence: the ball rises above the rim, exits sideways while
# still inside the rim band, and only afterwards is seen falling below the rim.
REPORTED_SEQUENCE_ROI = RimROI(90, 30, 110, 50)
REPORTED_SEQUENCE = [(25, 180), (25, 180), (25, 180), (50, 160), (75, 120), (90, 90),
                     (95, 65), (100, 20), (100, 15), (100, 35), (120, 55), (108, 70), (108, 85)]

# Bottom-boundary evidence only: the ball is seen above the rim and then below
# it one step later, with no intermediate observation inside the band.
SPARSE_MADE = [(25, 180), (25, 180), (25, 180), (85, 115), (90, 90),
               (95, 65), (100, 20), (100, 15), (100, 90), (100, 115)]

# Sideways exit inside the rim band on the way up: never a passage downwards.
SIDEWAYS_EXIT = [(25, 180), (25, 180), (25, 180), (85, 115), (90, 90),
                 (95, 65), (100, 35), (105, 20), (140, 25), (150, 45), (155, 75)]

WAYPOINT_MADE = [
    (25, 180, 0.25), (60, 150, 0.20), (85, 115, 0.18), (95, 70, 0.14),
    (100, 15, 0.14), (100, 40, 0.10), (102, 75, 0.16), (104, 110, 0.20),
]
WAYPOINT_MISS = [
    (25, 180, 0.25), (60, 150, 0.20), (85, 115, 0.18), (95, 70, 0.14),
    (100, 20, 0.14), (135, 55, 0.10), (150, 95, 0.16),
]
FRAME_RATES = [24, 30, 60, 120]


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


def test_sideways_exit_before_later_aligned_below_observations_is_not_made():
    """Regression: entering the ROI from above must not confirm a passage.

    The ball is briefly above the rim but leaves through the side while still
    inside the rim band, then reappears below the rim. No downward passage was
    ever observed, so the attempt cannot be a made candidate.
    """
    engine = EventAnalyzer(30, REPORTED_SEQUENCE_ROI)
    feed(engine, REPORTED_SEQUENCE)
    assert engine.summary()["made_candidates"] == 0
    assert [shot["outcome"] for shot in engine.shots] != ["made_candidate"]
    event_types = [event["type"] for event in engine.events if event["shot_id"] is not None]
    assert "made_shot_candidate" not in event_types


def test_sideways_exit_inside_the_rim_band_is_not_a_passage():
    engine = EventAnalyzer(30, REPORTED_SEQUENCE_ROI)
    feed(engine, SIDEWAYS_EXIT)
    engine.finalize(len(SIDEWAYS_EXIT) - 1)
    assert engine.summary()["made_candidates"] == 0


def test_sparse_sampling_can_still_confirm_a_passage():
    """A real passage may skip the interior; top and bottom must both be seen."""
    engine = EventAnalyzer(30, REPORTED_SEQUENCE_ROI)
    feed(engine, SPARSE_MADE)
    assert engine.shots[0]["outcome"] == "made_candidate"


def test_gap_between_above_and_below_cannot_invent_a_passage():
    engine = EventAnalyzer(30, ROI)
    feed(engine, MADE_PATH[:8] + [None] + MADE_PATH[9:])
    engine.finalize(len(MADE_PATH))
    assert engine.summary()["made_candidates"] == 0


def test_vertical_direction_uses_time_based_units():
    engine = EventAnalyzer(30, ROI)
    for frame, y in enumerate((200.0, 199.5, 199.0, 198.5)):
        engine._observe_motion(ball((100.0, y)), frame)
    # 1.5 px over 3 frames at 30 FPS is 15 px/s: real but below the limit.
    assert engine._vertical_speed() == pytest.approx(-15.0)
    assert engine._direction() is None

    climbing = EventAnalyzer(30, ROI)
    for frame, y in enumerate((200.0, 198.0, 196.0, 194.0)):
        climbing._observe_motion(ball((100.0, y)), frame)
    assert climbing._vertical_speed() == pytest.approx(-60.0)
    assert climbing._direction() == "rising"


def test_minimum_speed_means_the_same_physical_speed_at_every_frame_rate():
    """The old per-frame epsilon meant 24 px/s at 24 FPS but 120 px/s at 120 FPS."""
    for fps in FRAME_RATES:
        slow = EventAnalyzer(fps, ROI)
        for frame in range(6):
            slow._observe_motion(ball((100.0, 200.0 - (24.0 / fps) * frame)), frame)
        # 24 px/s is real motion but below the 36 px/s limit at every rate.
        assert slow._vertical_speed() == pytest.approx(-24.0, abs=1e-6)
        assert slow._direction() is None

        fast = EventAnalyzer(fps, ROI)
        for frame in range(6):
            fast._observe_motion(ball((100.0, 200.0 - (60.0 / fps) * frame)), frame)
        assert fast._vertical_speed() == pytest.approx(-60.0, abs=1e-6)
        assert fast._direction() == "rising"


def test_stationary_jitter_never_starts_an_attempt():
    """Sub-pixel alternation must not read as a rising ball at any frame rate."""
    for fps in FRAME_RATES:
        engine = EventAnalyzer(fps, ROI)
        for step in range(int(2.4 * fps)):
            x = 90.0 + min(step, 20)
            y = 180.0 + (0.5 if step % 2 else -0.5)
            engine.update(ball((x, y)), None, [PLAYER], step)
        engine.finalize(int(2.4 * fps) - 1)
        assert engine.shots == [], f"{fps} FPS produced an attempt from jitter"
        assert engine.events == [] or all(
            event["type"] == "possession_proxy_change" for event in engine.events
        )


@pytest.mark.parametrize("fps", FRAME_RATES)
def test_same_continuous_flight_is_judged_the_same_at_every_frame_rate(fps):
    """One physical trajectory, four capture rates, one outcome each."""
    factor, duration = 2, 4.0
    made = analyze_path(build_samples(WAYPOINT_MADE, factor), fps, duration)
    assert made.summary()["made_candidates"] == 1
    assert made.shots[0]["outcome"] == "made_candidate"

    missed = analyze_path(build_samples(WAYPOINT_MISS, factor), fps, duration)
    assert missed.shots[0]["outcome"] == "missed_candidate"


@pytest.mark.parametrize("fps", FRAME_RATES)
def test_missing_observations_do_not_manufacture_positions(fps):
    """Dropping the frames around the rim removes the evidence, not the ball."""
    factor, duration = 2, 4.0
    samples = build_samples(WAYPOINT_MADE, factor)
    steps = int(duration * fps)
    blanked = [
        step for step in range(steps)
        if (lambda point: ROI.x1 - 10 <= point[0] <= ROI.x2 + 10 and point[1] < ROI.y2)(
            at(samples, step / fps)
        )
    ]
    assert blanked, "the fixture must actually remove rim observations"
    result = analyze_path(samples, fps, duration, missing=tuple(blanked))
    assert result.summary()["made_candidates"] == 0
    assert result.summary()["unknown_outcomes"] == len(result.shots)
