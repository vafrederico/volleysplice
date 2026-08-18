#!/usr/bin/env python3
"""Audit suppression applied to only one production-ensemble component."""

from __future__ import annotations

import argparse
import csv
import io
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts" / "train-feedback-suppression-v3.py"
DEFAULT_PREVIOUS_OUTPUT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16/old-only-suppression-rally-audit.json"
)
DEFAULT_V2_OUTPUT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16/v2-only-suppression-rally-audit.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("previous-production", "all-labels-v2"),
        default="previous-production",
        help="The sole component from which suppression cuts are subtracted.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_PREVIOUS_OUTPUT if args.component == "previous-production" else DEFAULT_V2_OUTPUT
    )
    coverage_key = (
        "oldOnlySuppressionCoveredSeconds"
        if args.component == "previous-production"
        else "v2OnlySuppressionCoveredSeconds"
    )
    metric_key = "oldOnlySuppression" if args.component == "previous-production" else "v2OnlySuppression"

    ns = runpy.run_path(str(TRAIN_SCRIPT))
    interval = ns["Interval"]
    subtract = ns["subtract_intervals"]
    merge = ns["_merge_intervals"]
    total_duration = ns["_duration"]
    intersection_duration = ns["_intersection_duration"]
    pad_and_merge = ns["pad_and_merge_intervals"]
    item_type = ns["Item"]

    def ranges(path: Path) -> tuple[Any, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(interval(float(row["start"]), float(row["end"])) for row in payload["ranges"])

    old_heads, v2_heads = ns["load_heads"]()
    training_ids = {recording.id for recording in ns["load_manifest"](ns["ALL_LABELS_MANIFEST"]).recordings}
    split = json.loads((ns["DEFAULT_OUTPUT"] / "split-policy.json").read_text(encoding="utf-8"))
    feedback_fit = set(split["fitProjectIds"])
    entries = []
    for recording in ns["load_labeled_records"]().values():
        prepared = ns["prepare_recording"](recording, v2_heads.rally.feature_config, ns["FEATURE_CACHE"])
        provenance = "training-dataset" if recording.id in training_ids else "evaluation-validation-test-only"
        entries.append(item_type(prepared, provenance, None, True))
    for path in sorted(ns["FEEDBACK_ROOT"].glob("*/bundle.json")):
        prepared, _ = ns["feedback_prepared"](path, v2_heads.rally.feature_config)
        partition = "fit" if prepared.recording.id in feedback_fit else "held-out"
        entries.append(item_type(prepared, "export-feedback", partition, True))

    inference_root = ns["DEFAULT_OUTPUT"] / "inference"
    baseline_predictions: dict[str, tuple[Any, ...]] = {}
    candidate_predictions: dict[str, tuple[Any, ...]] = {}
    rallies: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        prepared = entry.prepared
        recording_id = prepared.recording.id
        baseline = ranges(inference_root / "current-production-ensemble" / f"{recording_id}.json")
        fully_suppressed = ranges(inference_root / "production-plus-suppression" / f"{recording_id}.json")
        removed = subtract(baseline, fully_suppressed)
        old = ns["predict_heads"](prepared, old_heads)
        v2 = ns["predict_heads"](prepared, v2_heads)
        candidate = (
            ns["union_intervals"](subtract(old, removed), v2)
            if args.component == "previous-production"
            else ns["union_intervals"](old, subtract(v2, removed))
        )
        baseline_predictions[recording_id] = baseline
        candidate_predictions[recording_id] = candidate

        baseline_padded = subtract(
            pad_and_merge(baseline, prepared.sequence.metadata.duration, 2.0, 3.0),
            prepared.recording.ignored_intervals,
        )
        candidate_padded = subtract(
            pad_and_merge(candidate, prepared.sequence.metadata.duration, 2.0, 3.0),
            prepared.recording.ignored_intervals,
        )
        for rally_number, core in enumerate(prepared.recording.rallies, start=1):
            evaluable_core = subtract((core,), prepared.recording.ignored_intervals)
            evaluable_duration = total_duration(evaluable_core)
            if evaluable_duration <= 1e-9:
                continue
            baseline_covered = intersection_duration(evaluable_core, baseline_padded)
            candidate_covered = intersection_duration(evaluable_core, candidate_padded)
            additional_missed = max(0.0, baseline_covered - candidate_covered)
            if additional_missed <= 1e-6:
                continue
            rallies.append({
                "file": prepared.recording.raw.get("sourceFilename", prepared.recording.video.name),
                "recordingId": recording_id,
                "provenance": entry.provenance,
                "feedbackPartition": entry.feedback_partition,
                "environment": prepared.recording.environment,
                "rallyNumber": rally_number,
                "coreStart": core.start,
                "coreEnd": core.end,
                "evaluableCoreDurationSeconds": evaluable_duration,
                "productionCoveredSeconds": baseline_covered,
                coverage_key: candidate_covered,
                "additionalMissedCoreSeconds": additional_missed,
                "fullyMissedAfterPadding": candidate_covered <= 1e-6,
            })
        print(f"Audited {index}/{len(entries)}: {recording_id}", flush=True)

    rallies.sort(key=lambda row: (
        not row["fullyMissedAfterPadding"],
        -row["additionalMissedCoreSeconds"],
        row["file"],
        row["rallyNumber"],
    ))

    def selected_metrics(selected: Iterable[Any], predictions: dict[str, tuple[Any, ...]]) -> dict[str, float]:
        metric = ns["metric_for"](list(selected), predictions, 2.0)
        return {key: metric[key] for key in ("P_pad", "R_core", "F1_padP_coreR")}

    grass = [entry for entry in entries if entry.prepared.recording.environment == "grass"]
    payload = {
        "schemaVersion": 1,
        "experiment": ns["EXPERIMENT_ID"],
        "suppressedComponent": args.component,
        "strategy": (
            "(previous production minus current suppression cuts) union unchanged all-labels v2"
            if args.component == "previous-production"
            else "unchanged previous production union (all-labels v2 minus current suppression cuts)"
        ),
        "paddingSecondsBeforeAndAfter": 2.0,
        "joinGapSecondsStrictlyLessThan": 3.0,
        "summary": {
            "ralliesWithAdditionalMiss": len(rallies),
            "fullyMissed": sum(row["fullyMissedAfterPadding"] for row in rallies),
            "partiallyMissed": sum(not row["fullyMissedAfterPadding"] for row in rallies),
            "additionalMissedCoreSeconds": sum(row["additionalMissedCoreSeconds"] for row in rallies),
            "grassAffected": sum(row["environment"] == "grass" for row in rallies),
            "grassFullyMissed": sum(row["environment"] == "grass" and row["fullyMissedAfterPadding"] for row in rallies),
            "grassAdditionalMissedCoreSeconds": sum(
                row["additionalMissedCoreSeconds"] for row in rallies if row["environment"] == "grass"
            ),
        },
        "metrics": {
            "all": {
                "production": selected_metrics(entries, baseline_predictions),
                metric_key: selected_metrics(entries, candidate_predictions),
            },
            "grass": {
                "production": selected_metrics(grass, baseline_predictions),
                metric_key: selected_metrics(grass, candidate_predictions),
            },
        },
        "rallies": rallies,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    ns["atomic_write_text"](output, json.dumps(payload, indent=2, allow_nan=False) + "\n")

    csv_output = output.with_suffix(".csv")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rallies[0]))
    writer.writeheader()
    writer.writerows(rallies)
    ns["atomic_write_text"](csv_output, buffer.getvalue())
    print(output)
    print(csv_output)


if __name__ == "__main__":
    main()
