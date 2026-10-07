"""Score V2 against human labels on real clips with matching clip separation.

These functions measure quality; they do not tune thresholds. A clip that
appears in both the tuning and evaluation sets is rejected, because a threshold
chosen on a clip and then scored on the same clip reports fit, not accuracy.
Nothing here reads frames or runs inference: it compares what the pipeline
already exported with labels a human wrote down.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Pipeline outcome names and the ground-truth outcomes they must be compared to.
PREDICTED_OUTCOMES = ("made_candidate", "missed_candidate", "unknown")

ANNOTATION_HELP = (
    "Clip annotation JSON needs: clip_id (string), fps (positive number), "
    "ball (one entry per frame) and shots (one entry per real attempt). "
    "Each ball entry is either true/false/null, or an object with 'visible' "
    "and optional 'bbox' [x1,y1,x2,y2] for true positives to be checked by IoU. "
    "Each shot is {release_frame, outcome: made|missed, outcome_frame?}. "
    "Mark 'human_labeled': true only for clips a person actually checked."
)


@dataclass(frozen=True)
class BallFrame:
    """One frame's human label: was the ball visible, and where?"""

    visible: bool
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ShotLabel:
    """One real attempt and its true outcome."""

    release_frame: int
    outcome: str
    outcome_frame: int | None = None


@dataclass(frozen=True)
class ClipAnnotation:
    """Human labels for one clip."""

    clip_id: str
    fps: float
    ball: tuple[BallFrame, ...]
    shots: tuple[ShotLabel, ...]
    human_labeled: bool = False
    source: str | None = None


@dataclass(frozen=True)
class ClipPrediction:
    """What the pipeline exported for one clip."""

    clip_id: str
    ball_track_ids: tuple[int | None, ...]
    shots: tuple[dict, ...]

    @property
    def ball_frames(self) -> tuple[bool, ...]:
        """A frame counts as detected only when a ball track was selected."""
        return tuple(identity is not None for identity in self.ball_track_ids)

    @classmethod
    def from_analytics(cls, clip_id: str, report: dict) -> "ClipPrediction":
        """Read the exported V2 analytics JSON; its schema is not modified here."""
        frames = int(report.get("video", {}).get("frames_processed", 0))
        trace = report.get("basketball", {}).get("ball_track_by_frame")
        if trace is None:
            raise ValueError(
                f"{clip_id}: analytics has no 'basketball.ball_track_by_frame'; "
                "re-run the pipeline with this version to export the per-frame ball trace."
            )
        if len(trace) < frames:
            raise ValueError(
                f"{clip_id}: ball trace has {len(trace)} entries for {frames} frames."
            )
        identities: list[int | None] = []
        for value in trace[:frames]:
            if value is None:
                identities.append(None)
            elif isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{clip_id}: ball trace entries must be integers or null.")
            else:
                identities.append(value)
        return cls(
            clip_id=clip_id,
            ball_track_ids=tuple(identities),
            shots=tuple(report.get("shot_candidates", {}).get("shots", [])),
        )


@dataclass
class EvaluationReport:
    """Measured quality on labeled clips. Every value is a count or ratio."""

    clips: list[str] = field(default_factory=list)
    frames: int = 0
    ball: dict = field(default_factory=dict)
    continuity: dict = field(default_factory=dict)
    shot_events: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clips": list(self.clips),
            "frames": self.frames,
            "ball": self.ball,
            "continuity": self.continuity,
            "shot_events": self.shot_events,
            "outcomes": self.outcomes,
            "limitations": list(self.limitations),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number.")
    return float(value)


def _nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value


def parse_ball_frame(value, index: int) -> BallFrame:
    """Accept true/false/null or an object with 'visible' and optional 'bbox'."""
    if value is None:
        return BallFrame(False)
    if isinstance(value, bool):
        return BallFrame(value)
    if isinstance(value, dict):
        if "visible" not in value:
            raise ValueError(f"ball[{index}] object needs a 'visible' field.")
        visible = value["visible"]
        if not isinstance(visible, bool):
            raise ValueError(f"ball[{index}]['visible'] must be true or false.")
        bbox = value.get("bbox")
        if bbox is None:
            return BallFrame(visible)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"ball[{index}]['bbox'] must be [x1,y1,x2,y2].")
        values = tuple(float(part) for part in bbox)
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"ball[{index}]['bbox'] must have x2 > x1 and y2 > y1.")
        return BallFrame(visible, values)
    raise ValueError(f"ball[{index}] must be true, false, null, or an object.")


def load_annotation(path: Path) -> ClipAnnotation:
    """Read one clip annotation and reject shapes the metrics cannot trust."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Annotation is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Annotation must be a JSON object: {path}")
    clip_id = payload.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError(f"Annotation {path} needs a non-empty string 'clip_id'.")
    fps = _positive_number(payload.get("fps"), f"{clip_id}: fps")
    raw_ball = payload.get("ball")
    if not isinstance(raw_ball, list) or not raw_ball:
        raise ValueError(f"{clip_id}: 'ball' must be a non-empty list, one entry per frame.")
    ball = tuple(parse_ball_frame(value, index) for index, value in enumerate(raw_ball))
    raw_shots = payload.get("shots", [])
    if not isinstance(raw_shots, list):
        raise ValueError(f"{clip_id}: 'shots' must be a list.")
    shots: list[ShotLabel] = []
    for index, entry in enumerate(raw_shots):
        if not isinstance(entry, dict):
            raise ValueError(f"{clip_id}: shots[{index}] must be an object.")
        outcome = entry.get("outcome")
        if outcome not in ("made", "missed"):
            raise ValueError(f"{clip_id}: shots[{index}]['outcome'] must be 'made' or 'missed'.")
        release = _nonnegative_int(entry.get("release_frame"), f"{clip_id}: shots[{index}]['release_frame']")
        outcome_frame = entry.get("outcome_frame")
        if outcome_frame is not None:
            _nonnegative_int(outcome_frame, f"{clip_id}: shots[{index}]['outcome_frame']")
        shots.append(ShotLabel(release, outcome, outcome_frame))
    human_labeled = payload.get("human_labeled", False)
    if not isinstance(human_labeled, bool):
        raise ValueError(f"{clip_id}: 'human_labeled' must be true or false.")
    source = payload.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError(f"{clip_id}: 'source' must be a string when present.")
    return ClipAnnotation(clip_id, fps, ball, tuple(shots), human_labeled, source)


def ensure_disjoint(tuning_ids: Iterable[str], evaluation_ids: Iterable[str]) -> None:
    """Refuse to evaluate on a clip that was also used to choose thresholds."""
    tuning = {str(value) for value in tuning_ids}
    evaluation = {str(value) for value in evaluation_ids}
    overlap = sorted(tuning & evaluation)
    if overlap:
        raise ValueError(
            "Evaluation clips must be separate from tuning clips; also in the "
            f"tuning set: {', '.join(overlap)}"
        )


def box_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float],
) -> float:
    """Intersection over union of two xyxy boxes; 0.0 when they do not overlap."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _ratio(numerator: int, denominator: int) -> float | None:
    """Undefined ratios stay None rather than becoming a misleading 0.0."""
    return numerator / denominator if denominator else None


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def ball_metrics(
    annotations: list[ClipAnnotation], predictions: list[ClipPrediction],
) -> tuple[dict, "list[str]"]:
    """Frame-level ball precision/recall against human visibility labels."""
    true_positive = false_positive = false_negative = 0
    true_negative = 0
    labels_with_boxes = 0
    for annotation, prediction in zip(annotations, predictions):
        if len(annotation.ball) != len(prediction.ball_track_ids):
            raise ValueError(
                f"{annotation.clip_id}: {len(annotation.ball)} labeled frames but "
                f"{len(prediction.ball_track_ids)} predicted frames; labels must cover the clip."
            )
        labels_with_boxes += sum(label.bbox is not None for label in annotation.ball)
        for index, label in enumerate(annotation.ball):
            detected = prediction.ball_track_ids[index] is not None
            if label.visible and detected:
                true_positive += 1
            elif detected:
                false_positive += 1
            elif label.visible:
                false_negative += 1
            else:
                true_negative += 1
    metrics = _prf(true_positive, false_positive, false_negative)
    metrics["true_negatives"] = true_negative
    metrics["frame_accuracy"] = _ratio(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
    )
    metrics["labeled_boxes"] = labels_with_boxes
    metrics["box_localization_rate"] = None
    notes = [
        "Precision/recall are frame-level presence: a selected ball counts as a "
        "true positive even if it is the wrong object, because the pipeline does "
        "not export per-frame boxes. Localization needs a separately exported "
        "box trace and is reported as undefined rather than assumed perfect.",
        "Simultaneous ground-truth balls are not distinguished; one selected ball "
        "per frame is scored.",
    ]
    return metrics, notes


def continuity_metrics(
    annotations: list[ClipAnnotation], predictions: list[ClipPrediction],
) -> tuple[dict, "list[str]"]:
    """Tracking continuity over adjacent labeled-visible frame pairs."""
    pairs = 0
    one_track = 0
    gaps = 0
    restarts = 0
    frames = 0
    for annotation, prediction in zip(annotations, predictions):
        identities = prediction.ball_track_ids
        frames += len(identities)
        for index in range(1, len(annotation.ball)):
            if not (annotation.ball[index - 1].visible and annotation.ball[index].visible):
                continue
            pairs += 1
            previous, current = identities[index - 1], identities[index]
            if previous is not None and previous == current:
                one_track += 1
            elif previous is None and current is None:
                gaps += 1
            else:
                restarts += 1
    return {
        "visible_frame_pairs": pairs,
        "same_id_pairs": one_track,
        "undetected_pairs": gaps,
        "id_restarts": restarts,
        "continuity_rate": _ratio(one_track, pairs),
        "frames": frames,
    }, [
        "Continuity is counted only where both frames are human-labeled visible, "
        "so a detection failure is reported as a missed detection, not as a break.",
    ]


def _overlaps(prediction: dict, label: ShotLabel, tolerance: int) -> bool:
    start = int(prediction.get("start_frame", -1))
    decision = int(prediction.get("decision_frame", start))
    return start <= label.release_frame + tolerance and decision >= label.release_frame - tolerance


def _match_shots(
    labels: list[ShotLabel], predictions: list[dict], tolerance: int,
) -> tuple[list[tuple[ShotLabel, dict]], int, int]:
    """Greedy one-to-one match by release proximity; returns matches, FP, FN."""
    remaining = list(range(len(predictions)))
    matches: list[tuple[ShotLabel, dict]] = []
    for label in sorted(labels, key=lambda item: item.release_frame):
        candidates = [
            index for index in remaining
            if _overlaps(predictions[index], label, tolerance)
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: abs(int(predictions[index].get("start_frame", 0)) - label.release_frame))
        matches.append((label, predictions[best]))
        remaining.remove(best)
    return matches, len(remaining), len(labels) - len(matches)


def shot_event_metrics(
    annotations: list[ClipAnnotation], predictions: list[ClipPrediction], tolerance_frames: int = 0,
) -> tuple[dict, "list[str]"]:
    """Precision/recall for attempts and for made/missed outcome events."""
    attempts_tp = attempts_fp = attempts_fn = 0
    made_tp = made_fp = made_fn = 0
    missed_tp = missed_fp = missed_fn = 0
    for annotation, prediction in zip(annotations, predictions):
        matched, unmatched_predictions, unmatched_labels = _match_shots(
            list(annotation.shots), list(prediction.shots), tolerance_frames,
        )
        attempts_tp += len(matched)
        attempts_fp += unmatched_predictions
        attempts_fn += unmatched_labels
        matched_ids = {id(item[1]) for item in matched}
        for label, shot in matched:
            predicted = shot.get("outcome")
            if label.outcome == "made":
                if predicted == "made_candidate":
                    made_tp += 1
                elif predicted in ("missed_candidate", "unknown"):
                    made_fn += 1
            else:
                if predicted == "missed_candidate":
                    missed_tp += 1
                elif predicted in ("made_candidate", "unknown"):
                    missed_fn += 1
        for shot in prediction.shots:
            if id(shot) in matched_ids:
                continue
            if shot.get("outcome") == "made_candidate":
                made_fp += 1
            elif shot.get("outcome") == "missed_candidate":
                missed_fp += 1
    return {
        "tolerance_frames": tolerance_frames,
        "attempts": _prf(attempts_tp, attempts_fp, attempts_fn),
        "made_events": _prf(made_tp, made_fp, made_fn),
        "missed_events": _prf(missed_tp, missed_fp, missed_fn),
    }, [
        "An unmatched attempt is a false positive whether it was a hallucinated "
        "shot or a real shot placed outside the frame tolerance.",
    ]


def outcome_error_metrics(
    annotations: list[ClipAnnotation], predictions: list[ClipPrediction], tolerance_frames: int = 0,
) -> tuple[dict, "list[str]"]:
    """Outcome correctness on matched attempts, with unknown counted separately."""
    correct = wrong = unknown = 0
    confusion = {f"{truth}->{predicted}": 0 for truth in ("made", "missed") for predicted in PREDICTED_OUTCOMES}
    for annotation, prediction in zip(annotations, predictions):
        matched, _, _ = _match_shots(list(annotation.shots), list(prediction.shots), tolerance_frames)
        for label, shot in matched:
            predicted = shot.get("outcome")
            if predicted not in PREDICTED_OUTCOMES:
                raise ValueError(
                    f"{annotation.clip_id}: unknown predicted outcome {predicted!r}; "
                    f"expected one of {', '.join(PREDICTED_OUTCOMES)}."
                )
            confusion[f"{label.outcome}->{predicted}"] += 1
            if predicted == "unknown":
                unknown += 1
            elif predicted == f"{label.outcome}_candidate":
                correct += 1
            else:
                wrong += 1
    decided = correct + wrong
    return {
        "matched_attempts": correct + wrong + unknown,
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "outcome_accuracy_on_decided": _ratio(correct, decided),
        "unknown_rate": _ratio(unknown, correct + wrong + unknown),
        "confusion": confusion,
    }, [
        "Outcome error is reported on matched attempts only; a wrong outcome on "
        "an unmatched attempt is counted as a false-positive event instead.",
        "Unknown is never folded into accuracy: an abstention is not a correct answer.",
    ]


def evaluate(
    annotations: list[ClipAnnotation], predictions: list[ClipPrediction],
    tolerance_frames: int = 0,
) -> EvaluationReport:
    """Measure every axis on the same labeled clips."""
    _require(len(annotations) == len(predictions), "Each annotation needs one prediction.")
    _require(bool(annotations), "At least one labeled clip is required.")
    for annotation, prediction in zip(annotations, predictions):
        _require(
            annotation.clip_id == prediction.clip_id,
            f"Clip mismatch: annotation {annotation.clip_id!r} vs prediction {prediction.clip_id!r}.",
        )
    ball, ball_notes = ball_metrics(annotations, predictions)
    continuity, continuity_notes = continuity_metrics(annotations, predictions)
    shot_events, shot_notes = shot_event_metrics(annotations, predictions, tolerance_frames)
    outcomes, outcome_notes = outcome_error_metrics(annotations, predictions, tolerance_frames)
    limitations = ball_notes + continuity_notes + shot_notes + outcome_notes
    if not all(annotation.human_labeled for annotation in annotations):
        unlabeled = [annotation.clip_id for annotation in annotations if not annotation.human_labeled]
        limitations.append(
            "These clips are not marked 'human_labeled'; the numbers describe "
            f"label agreement, not real accuracy: {', '.join(unlabeled)}"
        )
    return EvaluationReport(
        clips=[annotation.clip_id for annotation in annotations],
        frames=sum(len(annotation.ball) for annotation in annotations),
        ball=ball,
        continuity=continuity,
        shot_events=shot_events,
        outcomes=outcomes,
        limitations=limitations,
    )


def format_report(report: EvaluationReport) -> str:
    """Render the measured numbers for a human, without inventing a headline."""
    def ratio(value) -> str:
        return "undefined" if value is None else f"{value:.3f}"

    ball = report.ball
    continuity = report.continuity
    outcomes = report.outcomes
    localization = ball["box_localization_rate"]
    box_note = (
        f"not measured ({ball['labeled_boxes']} labeled boxes)"
        if localization is None else ratio(localization)
    )
    lines = [
        f"Clips evaluated: {len(report.clips)} ({report.frames} labeled frames)",
        "",
        "Ball detection (frame level)",
        f"  precision {ratio(ball['precision'])}  recall {ratio(ball['recall'])}  f1 {ratio(ball['f1'])}",
        f"  tp {ball['true_positives']}  fp {ball['false_positives']}  fn {ball['false_negatives']}"
        f"  tn {ball['true_negatives']}",
        f"  box localization {box_note}",
        "",
        "Tracking continuity",
        f"  continuity {ratio(continuity['continuity_rate'])} over {continuity['visible_frame_pairs']} visible pairs",
        f"  undetected pairs {continuity['undetected_pairs']}  id restarts {continuity['id_restarts']}",
        "",
        "Shot events (frame tolerance "
        f"{report.shot_events['tolerance_frames']})",
    ]
    for name in ("attempts", "made_events", "missed_events"):
        entry = report.shot_events[name]
        lines.append(
            f"  {name:14s} precision {ratio(entry['precision'])}  recall {ratio(entry['recall'])}"
            f"  tp {entry['true_positives']} fp {entry['false_positives']} fn {entry['false_negatives']}"
        )
    lines += [
        "",
        "Outcome errors (matched attempts)",
        f"  accuracy on decided {ratio(outcomes['outcome_accuracy_on_decided'])}"
        f"  correct {outcomes['correct']}  wrong {outcomes['wrong']}  unknown {outcomes['unknown']}",
        f"  unknown rate {ratio(outcomes['unknown_rate'])}",
    ]
    for key in sorted(outcomes["confusion"]):
        lines.append(f"  {key}: {outcomes['confusion'][key]}")
    if report.limitations:
        lines += ["", "Limitations"]
        lines += [f"  - {note}" for note in report.limitations]
    return "\n".join(lines)
