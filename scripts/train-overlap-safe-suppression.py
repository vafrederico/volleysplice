#!/usr/bin/env python3
"""Retrain only the suppression head with overlap-safe feedback targets.

An exported false-positive interval can represent a split or boundary correction
rather than wholly invalid footage. If such an interval intersects any included
corrected rally core, this trainer excludes the entire interval from the
suppression-positive class, including samples that the production baseline also
selected. The production and all-labels-v2 component models remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts" / "train-feedback-suppression-v3.py"
DEFAULT_NAME = "suppression-overlap-exclusion-retrained"


def main() -> None:
    ns = runpy.run_path(str(TRAIN_SCRIPT))
    default_output = ns["DEFAULT_OUTPUT"] / "models" / DEFAULT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite suppression model {output}")

    interval = ns["Interval"]

    def intervals(rows: Iterable[dict[str, Any]]) -> tuple[Any, ...]:
        return tuple(
            interval(float(row["start"]), float(row["end"])) for row in rows
        )

    def overlaps(left: Any, right: Any) -> bool:
        return left.start < right.end and right.start < left.end

    original = ns["load_model"](
        ns["DEFAULT_OUTPUT"]
        / "models"
        / "v3-candidate2-four-head"
        / "suppression"
    )
    _, v2_heads = ns["load_heads"]()
    split = json.loads(
        (ns["DEFAULT_OUTPUT"] / "split-policy.json").read_text(encoding="utf-8")
    )
    fit_ids = set(split["fitProjectIds"])

    historical = [
        ns["prepare_recording"](
            recording,
            v2_heads.rally.feature_config,
            ns["FEATURE_CACHE"],
        )
        for recording in ns["load_manifest"](
            ns["ALL_LABELS_MANIFEST"]
        ).recordings
    ]
    feedback: list[Any] = []
    feedback_payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(ns["FEEDBACK_ROOT"].glob("*/bundle.json")):
        prepared, payload = ns["feedback_prepared"](
            path, v2_heads.rally.feature_config
        )
        if prepared.recording.id in fit_ids:
            feedback.append(prepared)
            feedback_payloads[prepared.recording.id] = payload
    training = [*historical, *feedback]

    train_values: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    per_recording: list[dict[str, Any]] = []
    excluded_intervals: list[dict[str, Any]] = []
    for item in training:
        recording_id = item.recording.id
        times = item.sequence.times
        truth = item.labels > 0.5
        baseline_payload = json.loads(
            (
                ns["DEFAULT_OUTPUT"]
                / "inference"
                / "current-production-ensemble"
                / f"{recording_id}.json"
            ).read_text(encoding="utf-8")
        )
        predicted = (
            ns["labels_for_times"](times, intervals(baseline_payload["ranges"]))
            > 0.5
        )
        explicit = ns["hard_negative_mask"](item)
        positive_before_exclusion = (
            item.sample_mask & ~truth & (predicted | explicit)
        )

        exclusion = np.zeros(len(times), dtype=bool)
        payload = feedback_payloads.get(recording_id)
        if payload is not None:
            corrections = payload["corrections"]
            truth_ranges = tuple(
                interval(float(row["coreStart"]), float(row["coreEnd"]))
                for row in corrections["correctedRanges"]
                if row.get("included") is True
            )
            false_positives = intervals(
                corrections["labels"].get("falsePositives", [])
            )
            suspect = tuple(
                false_positive
                for false_positive in false_positives
                if any(overlaps(false_positive, rally) for rally in truth_ranges)
            )
            exclusion = ns["labels_for_times"](times, suspect) > 0.5
            for false_positive in suspect:
                excluded_intervals.append(
                    {
                        "recordingId": recording_id,
                        "sourceFilename": item.recording.raw.get(
                            "sourceFilename", item.recording.video.name
                        ),
                        "start": false_positive.start,
                        "end": false_positive.end,
                    }
                )

        positive = positive_before_exclusion & ~exclusion
        negative = item.sample_mask & truth
        selected = positive | negative
        if not np.any(positive) or not np.any(negative):
            raise ValueError(
                f"corrected suppression target lacks a class for {recording_id}"
            )
        train_values.append(item.contextual_values[selected])
        train_labels.append(positive[selected].astype(np.float32))
        per_recording.append(
            {
                "id": recording_id,
                "positiveSuppressionSamplesBeforeExclusion": int(
                    np.sum(positive_before_exclusion)
                ),
                "positiveSuppressionSamples": int(np.sum(positive)),
                "excludedPositiveSamples": int(
                    np.sum(positive_before_exclusion & exclusion)
                ),
                "negativeRallySamples": int(np.sum(negative)),
            }
        )

    model = ns["train_logistic_model"](
        train_values,
        train_labels,
        [],
        [],
        original.feature_config,
        original.feature_names,
        original.decoder,
        ns["fixed_training"](
            int(v2_heads.rally.training_summary["bestEpoch"])
        ),
        prediction_task=ns["DEAD_STATE_TASK"],
    )
    model.feature_version = original.feature_version
    excluded_samples = sum(
        row["excludedPositiveSamples"] for row in per_recording
    )
    model.training_summary.update(
        {
            "experiment": ns["EXPERIMENT_ID"],
            "candidate": "false-positive-suppression-head-overlap-exclusion",
            "specialistRole": "veto selected dead/setup/side-switch activity",
            "positiveDefinition": (
                "valid non-rally samples selected by current production or explicit "
                "hard-negative labels, excluding complete false-positive intervals "
                "that intersect an included corrected rally core"
            ),
            "negativeDefinition": "valid human rally samples",
            "trainingRecordingIds": [
                item.recording.id for item in training
            ],
            "feedbackFeaturesReusedFromBundles": True,
            "overlapExclusionPolicy": (
                "Exclude the full half-open false-positive interval from the "
                "suppression-positive class when it has any positive-duration "
                "overlap with an included corrected rally core. Apply exclusion "
                "after baseline and explicit target union."
            ),
            "excludedFalsePositiveIntervals": excluded_intervals,
            "excludedPositiveSamples": excluded_samples,
            "perRecordingTargets": per_recording,
            "selectionPolicy": (
                "Training-label correction only; reuse the frozen production "
                "suppression decoder selected on the predeclared development scope."
            ),
            "productionEnsembleDecoderSelection": original.training_summary[
                "productionEnsembleDecoderSelection"
            ],
            "v3CandidateDecoderSelection": original.training_summary[
                "v3CandidateDecoderSelection"
            ],
            "parentSuppressionModelSha256": original.artifact_sha256,
        }
    )
    model.save(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "artifactSha256": model.artifact_sha256,
                "positiveSamples": model.training_summary["positiveSamples"],
                "negativeSamples": model.training_summary["negativeSamples"],
                "excludedIntervals": len(excluded_intervals),
                "excludedPositiveSamples": excluded_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
