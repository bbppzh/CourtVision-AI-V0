"""Metric arithmetic and clip-separation rules for real-clip evaluation."""

import json
from pathlib import Path

import pytest

from src.evaluation import (
    BallFrame,
    ClipAnnotation,
    ClipPrediction,
    ShotLabel,
    ball_metrics,
    box_iou,
    continuity_metrics,
    ensure_disjoint,
    evaluate,
    format_report,
    load_annotation,
    outcome_error_metrics,
    shot_event_metrics,
)

FPS = 30.0


def annotation(clip_id="clip", visible=(True, True, False, True), shots=(), **kwargs):
    return ClipAnnotation(
        clip_id=clip_id, fps=FPS,
        ball=tuple(BallFrame(flag) for flag in visible),
        shots=tuple(shots), **kwargs,
    )


def prediction(clip_id="clip", ids=(1, 1, None, 1), shots=()):
    return ClipPrediction(clip_id=clip_id, ball_track_ids=tuple(ids), shots=tuple(shots))


def test_ball_precision_recall_counts_each_frame_class():
    metrics, _ = ball_metrics([annotation()], [prediction()])
    assert metrics["true_positives"] == 3
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == pytest.approx(1.0)


def test_ball_false_positive_and_missed_detection_are_separate():
    # Labeled: ball visible in frame 0 only. Predicted: frames 1 and 2 only.
    metrics, _ = ball_metrics(
        [annotation(visible=(True, False, False))],
        [prediction(ids=(None, 7, 7))],
    )
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (0, 2, 1)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_undefined_ratios_stay_none_instead_of_zero():
    metrics, _ = ball_metrics([annotation(visible=(False, False))], [prediction(ids=(None, None))])
    assert metrics["precision"] is None and metrics["recall"] is None and metrics["f1"] is None
    assert metrics["true_negatives"] == 2


def test_wrong_frame_count_is_rejected():
    with pytest.raises(ValueError, match="must cover the clip"):
        ball_metrics([annotation(visible=(True, True))], [prediction(ids=(1,))])


def test_continuity_counts_only_pairs_visible_in_both_frames():
    metrics, _ = continuity_metrics(
        [annotation(visible=(True, True, False, True, True))],
        [prediction(ids=(1, 1, None, 2, 2))],
    )
    # Pairs (0,1) and (3,4) are labeled visible; pair (1,2) and (2,3) are not.
    assert metrics["visible_frame_pairs"] == 2
    assert metrics["same_id_pairs"] == 2
    assert metrics["id_restarts"] == 0
    assert metrics["continuity_rate"] == pytest.approx(1.0)


def test_continuity_reports_a_track_restart_the_camera_saw():
    metrics, _ = continuity_metrics(
        [annotation(visible=(True, True, True))],
        [prediction(ids=(1, 1, 2))],
    )
    assert metrics["id_restarts"] == 1
    assert metrics["continuity_rate"] == pytest.approx(0.5)


def test_continuity_distinguishes_a_missed_detection_from_a_break():
    metrics, _ = continuity_metrics(
        [annotation(visible=(True, True))],
        [prediction(ids=(None, None))],
    )
    assert metrics["undetected_pairs"] == 1
    assert metrics["continuity_rate"] == pytest.approx(0.0)


def test_shot_event_precision_and_recall_with_tolerance():
    attempt = {"shot_id": 1, "start_frame": 10, "decision_frame": 20, "outcome": "made_candidate"}
    labels = [ShotLabel(12, "made", 30)]
    exact, _ = shot_event_metrics([annotation(shots=labels)], [prediction(shots=(attempt,))], 0)
    assert exact["attempts"]["true_positives"] == 1
    assert exact["made_events"]["precision"] == pytest.approx(1.0)

    early = {"shot_id": 2, "start_frame": 40, "decision_frame": 50, "outcome": "made_candidate"}
    tolerant, _ = shot_event_metrics([annotation(shots=labels)], [prediction(shots=(early,))], 30)
    assert tolerant["attempts"]["true_positives"] == 1
    strict, _ = shot_event_metrics([annotation(shots=labels)], [prediction(shots=(early,))], 0)
    assert strict["attempts"]["false_positives"] == 1
    assert strict["attempts"]["false_negatives"] == 1


def test_a_labeled_shot_with_no_attempt_is_a_false_negative():
    metrics, _ = shot_event_metrics(
        [annotation(shots=[ShotLabel(5, "missed", 40)])], [prediction(shots=())],
    )
    assert metrics["attempts"]["false_negatives"] == 1
    assert metrics["attempts"]["precision"] is None


def test_outcome_errors_keep_unknown_out_of_accuracy():
    labels = [ShotLabel(10, "made", 20)]
    unknown = {"shot_id": 1, "start_frame": 10, "decision_frame": 15, "outcome": "unknown"}
    metrics, _ = outcome_error_metrics([annotation(shots=labels)], [prediction(shots=(unknown,))], 0)
    assert metrics["unknown"] == 1 and metrics["correct"] == 0 and metrics["wrong"] == 0
    assert metrics["outcome_accuracy_on_decided"] is None
    assert metrics["unknown_rate"] == pytest.approx(1.0)


def test_outcome_confusion_records_the_direction_of_the_error():
    labels = [ShotLabel(10, "made", 20)]
    wrong = {"shot_id": 1, "start_frame": 10, "decision_frame": 15, "outcome": "missed_candidate"}
    metrics, _ = outcome_error_metrics([annotation(shots=labels)], [prediction(shots=(wrong,))], 0)
    assert metrics["confusion"]["made->missed_candidate"] == 1
    assert metrics["outcome_accuracy_on_decided"] == pytest.approx(0.0)


def test_unmatched_attempt_counts_as_a_false_positive_event():
    attempt = {"shot_id": 1, "start_frame": 10, "decision_frame": 15, "outcome": "made_candidate"}
    metrics, _ = shot_event_metrics([annotation(shots=[])], [prediction(shots=(attempt,))], 0)
    assert metrics["made_events"]["false_positives"] == 1
    assert metrics["attempts"]["precision"] == pytest.approx(0.0)


def test_evaluate_reports_every_axis_and_its_limitations():
    labels = [ShotLabel(0, "made", 2)]
    made = {"shot_id": 1, "start_frame": 0, "decision_frame": 2, "outcome": "made_candidate"}
    report = evaluate(
        [annotation("a", visible=(True, True), shots=labels, human_labeled=True)],
        [prediction("a", ids=(1, 1), shots=(made,))],
    )
    payload = report.to_dict()
    assert payload["clips"] == ["a"] and payload["frames"] == 2
    assert payload["ball"]["precision"] == pytest.approx(1.0)
    assert payload["outcomes"]["correct"] == 1
    assert payload["outcomes"]["outcome_accuracy_on_decided"] == pytest.approx(1.0)
    assert payload["continuity"]["continuity_rate"] == pytest.approx(1.0)
    assert payload["limitations"]
    assert "precision" in format_report(report)


def test_unlabeled_clips_are_flagged_in_the_report():
    report = evaluate(
        [annotation("a", visible=(True,), human_labeled=False)],
        [prediction("a", ids=(1,))],
    )
    assert any("not marked 'human_labeled'" in note for note in report.limitations)


def test_mismatched_clip_ids_are_rejected():
    with pytest.raises(ValueError, match="Clip mismatch"):
        evaluate([annotation("a")], [prediction("b")])


def test_evaluation_clips_must_be_separate_from_tuning_clips():
    with pytest.raises(ValueError, match="separate from tuning clips"):
        ensure_disjoint(["held_out"], ["held_out", "other"])
    with pytest.raises(ValueError, match="separate from tuning clips"):
        ensure_disjoint(["sample", "sample"], ["sample"])
    ensure_disjoint(["tuned"], ["fresh"])
    ensure_disjoint([], ["fresh"])


def test_box_iou_bounds():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def write_annotation(tmp_path, payload, name="labels.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_annotation_accepts_booleans_objects_and_null(tmp_path):
    path = write_annotation(tmp_path, {
        "clip_id": "a", "fps": 30, "human_labeled": True,
        "ball": [True, {"visible": True, "bbox": [1, 2, 5, 6]}, None, {"visible": False}],
        "shots": [{"release_frame": 3, "outcome": "missed"}],
    })
    loaded = load_annotation(path)
    assert loaded.clip_id == "a" and loaded.human_labeled
    assert [frame.visible for frame in loaded.ball] == [True, True, False, False]
    assert loaded.ball[1].bbox == (1.0, 2.0, 5.0, 6.0)
    assert loaded.shots[0].outcome == "missed"


@pytest.mark.parametrize("payload", [
    {"clip_id": "", "fps": 30, "ball": [True]},
    {"clip_id": "a", "fps": 0, "ball": [True]},
    {"clip_id": "a", "fps": 30, "ball": []},
    {"clip_id": "a", "fps": 30, "ball": ["yes"]},
    {"clip_id": "a", "fps": 30, "ball": [{"visible": "yes"}]},
    {"clip_id": "a", "fps": 30, "ball": [{"visible": True, "bbox": [1, 2, 3]}]},
    {"clip_id": "a", "fps": 30, "ball": [True], "shots": [{"release_frame": 1, "outcome": "maybe"}]},
    {"clip_id": "a", "fps": 30, "ball": [True], "shots": [{"outcome": "made"}]},
])
def test_load_annotation_rejects_untrustworthy_labels(tmp_path, payload):
    with pytest.raises(ValueError):
        load_annotation(write_annotation(tmp_path, payload))


def test_prediction_reads_the_exported_per_frame_ball_trace():
    report = {
        "video": {"frames_processed": 4},
        "basketball": {"ball_track_by_frame": [1, 1, None, 2]},
        "shot_candidates": {"shots": []},
    }
    parsed = ClipPrediction.from_analytics("a", report)
    assert parsed.ball_track_ids == (1, 1, None, 2)
    assert parsed.ball_frames == (True, True, False, True)


def test_prediction_rejects_a_missing_or_short_trace():
    with pytest.raises(ValueError, match="ball_track_by_frame"):
        ClipPrediction.from_analytics("a", {"video": {"frames_processed": 2}, "basketball": {}})
    with pytest.raises(ValueError, match="entries for"):
        ClipPrediction.from_analytics(
            "a", {"video": {"frames_processed": 5}, "basketball": {"ball_track_by_frame": [1, None]}},
        )


def test_detection_rate_is_not_reported_as_accuracy():
    """A high visible-ball rate must not leak into the metric names or output."""
    report = evaluate(
        [annotation("a", visible=(True, True, True, True), human_labeled=True)],
        [prediction("a", ids=(1, 1, 1, 1))],
    )
    text = format_report(report)
    assert "detection_rate" not in text
    for forbidden in ("accuracy" if False else "visible-ball rate", "detection rate"):
        assert forbidden not in text
    assert report.to_dict()["ball"]["precision"] == pytest.approx(1.0)


def write_analytics(tmp_path, frames, trace, shots=()):
    path = tmp_path / "clip_tracking.json"
    path.write_text(json.dumps({
        "video": {"frames_processed": frames},
        "basketball": {"ball_track_by_frame": list(trace)},
        "shot_candidates": {"shots": list(shots)},
    }), encoding="utf-8")
    return path


def test_cli_refuses_a_clip_that_is_both_tuning_and_evaluation(tmp_path, capsys):
    import evaluate_v2

    labels = write_annotation(tmp_path, {
        "clip_id": "sample", "fps": 30, "human_labeled": True, "ball": [True, False], "shots": [],
    })
    analytics = write_analytics(tmp_path, 2, [1, None])
    status = evaluate_v2.main([
        "--predictions", str(analytics), "--labels", str(labels), "--tuning-clips", "sample",
    ])
    assert status == 1
    assert "separate from tuning clips" in capsys.readouterr().err


def test_cli_reads_tuning_clips_from_a_manifest(tmp_path, capsys):
    import evaluate_v2

    labels = write_annotation(tmp_path, {
        "clip_id": "sample", "fps": 30, "human_labeled": True, "ball": [True, False], "shots": [],
    })
    analytics = write_analytics(tmp_path, 2, [1, None])
    manifest = write_annotation(tmp_path, {"tuning_clips": ["sample"]}, name="manifest.json")
    status = evaluate_v2.main([
        "--predictions", str(analytics), "--labels", str(labels),
        "--tuning-manifest", str(manifest),
    ])
    assert status == 1
    assert "separate from tuning clips" in capsys.readouterr().err


def test_cli_scores_a_labeled_clip_and_writes_the_report(tmp_path, capsys):
    import evaluate_v2

    labels = write_annotation(tmp_path, {
        "clip_id": "eval_a", "fps": 30, "human_labeled": True,
        "ball": [True, True, False], "shots": [{"release_frame": 0, "outcome": "made"}],
    })
    attempt = {"shot_id": 1, "start_frame": 0, "decision_frame": 1, "outcome": "made_candidate"}
    analytics = write_analytics(tmp_path, 3, [1, 1, None], [attempt])
    output = tmp_path / "report.json"
    status = evaluate_v2.main([
        "--predictions", str(analytics), "--labels", str(labels),
        "--tuning-clips", "sample", "--output", str(output),
    ])
    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ball"]["precision"] == pytest.approx(1.0)
    assert payload["outcomes"]["correct"] == 1
    assert payload["tuning_clips"] == ["sample"]
    assert "precision" in capsys.readouterr().out


def test_cli_requires_human_labels_unless_allowed(tmp_path, capsys):
    import evaluate_v2

    labels = write_annotation(tmp_path, {
        "clip_id": "eval_a", "fps": 30, "ball": [True], "shots": [],
    })
    analytics = write_analytics(tmp_path, 1, [1])
    arguments = ["--predictions", str(analytics), "--labels", str(labels)]
    assert evaluate_v2.main(arguments) == 1
    assert "human_labeled" in capsys.readouterr().err
    assert evaluate_v2.main([*arguments, "--allow-unlabeled"]) == 0


def test_cli_rejects_mismatched_prediction_and_label_counts(tmp_path, capsys):
    import evaluate_v2

    labels = write_annotation(tmp_path, {
        "clip_id": "eval_a", "fps": 30, "human_labeled": True, "ball": [True], "shots": [],
    })
    status = evaluate_v2.main([
        "--predictions", str(write_analytics(tmp_path, 1, [1])), str(tmp_path / "second.json"),
        "--labels", str(labels),
    ])
    assert status == 1
    assert "one to one" in capsys.readouterr().err
