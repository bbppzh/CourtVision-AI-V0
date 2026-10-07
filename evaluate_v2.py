"""Score exported V2 analytics against human labels on real clips.

This is a measurement tool, not a tuning tool. It refuses to run when a clip
appears in both the tuning set and the evaluation set, because a threshold
chosen on a clip and then scored on that same clip measures fit, not accuracy.

Typical use:

    # 1. Run the pipeline on held-out clips (nothing here needs the network).
    python main.py --input eval/clips/a.mp4 --output out/a.mp4 --rim-roi 540,193,582,213

    # 2. Label each clip by hand and write eval/labels/a.json (see ANNOTATION_HELP).

    # 3. Score, naming the clips that were used to choose thresholds.
    python evaluate_v2.py \
        --predictions out/a_tracking.json eval/labels/... \
        --labels eval/labels/a.json \
        --tuning-clips sample_video \
        --output eval/report.json

Exit status is 0 when every requested metric was computed, 1 when the input is
invalid or a clip is not human-labeled, and 2 when no metric could be computed.
"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluation import (
    ANNOTATION_HELP,
    ClipPrediction,
    ensure_disjoint,
    evaluate,
    format_report,
    load_annotation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure V2 ball tracking and shot events against human labels.",
        epilog=ANNOTATION_HELP,
    )
    parser.add_argument(
        "--predictions", type=Path, nargs="+", required=True,
        help="One or more *_tracking.json files exported by the pipeline",
    )
    parser.add_argument(
        "--labels", type=Path, nargs="+", required=True,
        help="Matching human-label JSON files, one per prediction",
    )
    parser.add_argument(
        "--tuning-clips", nargs="*", default=[],
        help="Clip IDs that were used to choose thresholds; scoring them here is refused",
    )
    parser.add_argument(
        "--tuning-manifest", type=Path,
        help="JSON file with a 'tuning_clips' list, as an alternative to --tuning-clips",
    )
    parser.add_argument("--tolerance-frames", type=int, default=0,
                        help="Frames of slack when matching a predicted attempt to a labeled shot")
    parser.add_argument("--output", type=Path, help="Write the report as JSON")
    parser.add_argument("--allow-unlabeled", action="store_true",
                        help="Score clips whose labels are not marked 'human_labeled' (reported as label agreement only)")
    return parser.parse_args(argv)


def _load_tuning_ids(args: argparse.Namespace) -> set[str]:
    tuning = set(args.tuning_clips)
    if args.tuning_manifest is not None:
        try:
            payload = json.loads(args.tuning_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read tuning manifest {args.tuning_manifest}: {exc}") from exc
        listed = payload.get("tuning_clips") if isinstance(payload, dict) else None
        if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
            raise ValueError("Tuning manifest needs a 'tuning_clips' list of strings.")
        tuning |= set(listed)
    return tuning


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tolerance_frames < 0:
        print("Error: --tolerance-frames must be nonnegative.", file=sys.stderr)
        return 1
    if len(args.predictions) != len(args.labels):
        print(
            f"Error: {len(args.predictions)} predictions but {len(args.labels)} label files; "
            "they must pair one to one.",
            file=sys.stderr,
        )
        return 1
    try:
        tuning = _load_tuning_ids(args)
        annotations = [load_annotation(path) for path in args.labels]
        predictions = []
        for path, annotation in zip(args.predictions, annotations):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read analytics {path}: {exc}") from exc
            predictions.append(ClipPrediction.from_analytics(annotation.clip_id, report))
        ensure_disjoint(tuning, [annotation.clip_id for annotation in annotations])
        if not args.allow_unlabeled:
            unlabeled = [item.clip_id for item in annotations if not item.human_labeled]
            if unlabeled:
                print(
                    "Error: these clips are not marked 'human_labeled': "
                    f"{', '.join(unlabeled)}. Label them by hand, or pass "
                    "--allow-unlabeled to report label agreement only.",
                    file=sys.stderr,
                )
                return 1
        result = evaluate(annotations, predictions, args.tolerance_frames)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_report(result))
    payload = result.to_dict()
    payload["tuning_clips"] = sorted(tuning)
    payload["tolerance_frames"] = args.tolerance_frames
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
