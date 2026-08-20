#!/usr/bin/env python3
"""Train and evaluate the first reviewed-marker side-switch specialist.

Model-family selection uses grouped out-of-fold predictions from declared training
recordings.  The operating threshold is selected once on validation recordings.
Challenge, non-training/raw, and test recordings are opened only for the final
evaluation report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_specialist import (
    ELIGIBLE_ENVIRONMENTS,
    EVALUATION_SPLITS,
    FEATURE_SETS,
    MODEL_KIND,
    MODEL_SCHEMA_VERSION,
    ReviewedEvent,
    binary_metrics,
    event_vector,
    fit_specialist,
    grouped_cross_fit,
    labels_for,
    matrix_for,
    model_fingerprint,
    reviewed_events,
    select_threshold,
)
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings


DEFAULT_ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_REPORT = (
    DEFAULT_ROOT
    / "reports/side-switch/appearance-diagnostic-full-nas-v1.json"
)
DEFAULT_DECISIONS = (
    DEFAULT_ROOT
    / "reports/side-switch/appearance-review-decisions-full-nas-v1.json"
)
DEFAULT_MODEL_DIR = DEFAULT_ROOT / "models/side-switch-specialist-v1"
DEFAULT_EVALUATION = (
    DEFAULT_ROOT
    / "reports/side-switch/side-switch-specialist-v1-evaluation.json"
)
L2_GRID = (0.01, 0.1, 1.0, 10.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _row_counts(rows: Sequence[ReviewedEvent]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "switch": sum(row.label for row in rows),
        "noSwitch": sum(1 - row.label for row in rows),
        "recordings": len({row.recording_id for row in rows}),
        "recordingIds": sorted({row.recording_id for row in rows}),
        "sourceGroups": sorted({row.source_group for row in rows}),
    }


def _metric_bundle(
    rows: Sequence[ReviewedEvent], probabilities: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    metrics = binary_metrics(labels_for(rows), predicted)
    labels = labels_for(rows).astype(np.int64)
    positive_scores = probabilities[labels == 1]
    negative_scores = probabilities[labels == 0]
    if len(positive_scores) and len(negative_scores):
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        roc_auc = float(
            (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
            / comparisons.size
        )
    else:
        roc_auc = None
    order = np.argsort(-probabilities, kind="stable")
    ranked_labels = labels[order]
    if np.sum(ranked_labels):
        precision_at_rank = np.cumsum(ranked_labels) / np.arange(1, len(labels) + 1)
        average_precision = float(
            np.sum(precision_at_rank * ranked_labels) / np.sum(ranked_labels)
        )
    else:
        average_precision = None
    return {
        **metrics,
        "rocAuc": roc_auc,
        "averagePrecision": average_precision,
        "meanSwitchScore": (
            float(np.mean(positive_scores)) if len(positive_scores) else None
        ),
        "meanNoSwitchScore": (
            float(np.mean(negative_scores)) if len(negative_scores) else None
        ),
    }


def _group_metrics(
    rows: Sequence[ReviewedEvent],
    probabilities: np.ndarray,
    predicted: np.ndarray,
    key: Callable[[ReviewedEvent], str],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in sorted({key(row) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if key(row) == group],
            dtype=np.int64,
        )
        subset = [rows[index] for index in indices]
        values[group] = _metric_bundle(
            subset, probabilities[indices], predicted[indices]
        )
    return values


def _evaluation(
    rows: Sequence[ReviewedEvent], probabilities: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    return {
        "overall": _metric_bundle(rows, probabilities, predicted),
        "bySplit": _group_metrics(
            rows, probabilities, predicted, lambda row: row.split
        ),
        "byEnvironment": _group_metrics(
            rows, probabilities, predicted, lambda row: row.environment
        ),
        "byRecording": _group_metrics(
            rows, probabilities, predicted, lambda row: row.recording_id
        ),
        "bySourceType": _group_metrics(
            rows, probabilities, predicted, lambda row: row.source_type
        ),
        "byTargetStatus": _group_metrics(
            rows, probabilities, predicted, lambda row: row.target_status
        ),
    }


def _rank_candidate(
    candidate: Mapping[str, Any],
) -> tuple[float, float, float, int, float]:
    metrics = candidate["outOfFold"]["selectedThresholdMetrics"]
    return (
        float(metrics["f1"] or 0.0),
        float(metrics["precision"] or 0.0),
        float(metrics["recall"] or 0.0),
        -len(FEATURE_SETS[str(candidate["featureSet"])]),
        float(candidate["l2"]),
    )


def _dataset_row(row: ReviewedEvent, feature_set: str, role: str) -> dict[str, Any]:
    vector = event_vector(row.event, feature_set)
    return {
        "eventId": row.event_id,
        "recordingId": row.recording_id,
        "sourceGroup": row.source_group,
        "environment": row.environment,
        "sourceSplit": row.split,
        "role": role,
        "sourceType": row.source_type,
        "targetStatus": row.target_status,
        "transitionTime": row.event.get("transitionTime"),
        "decision": row.decision,
        "label": row.label,
        "features": [_finite_or_none(value) for value in vector],
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.appearance_report.resolve()
    decision_path = args.decisions.resolve()
    model_dir = args.model_dir.resolve()
    evaluation_path = args.evaluation_output.resolve()
    if model_dir.exists():
        raise FileExistsError(f"refusing to reuse model directory: {model_dir}")
    if evaluation_path.exists():
        raise FileExistsError(
            f"refusing to overwrite evaluation artifact: {evaluation_path}"
        )

    report = _load_json(report_path)
    decisions = _load_json(decision_path)
    rows, review_counts = reviewed_events(report, decisions)
    raw_decisions = decisions.get("decisions", {})
    if review_counts["missing"] or len(raw_decisions) != review_counts["events"]:
        raise ValueError(
            "training requires one saved review decision for every generated marker"
        )

    eligible_train_rows = [
        row
        for row in rows
        if row.split == "train" and row.environment in {"beach", "grass"}
    ]
    excluded_fit_recording_ids = tuple(sorted(set(args.exclude_fit_recording)))
    unknown_exclusions = sorted(
        set(excluded_fit_recording_ids)
        - {row.recording_id for row in eligible_train_rows}
    )
    if unknown_exclusions:
        raise ValueError(
            "fit exclusions are not eligible training recordings: "
            + ", ".join(unknown_exclusions)
        )
    train_rows = [
        row
        for row in eligible_train_rows
        if row.recording_id not in excluded_fit_recording_ids
    ]
    if excluded_fit_recording_ids:
        validate_side_switch_fit_recordings(
            tuple(sorted({row.recording_id for row in train_rows}))
        )
    validation_rows = [
        row
        for row in rows
        if row.split == "validation" and row.environment in ELIGIBLE_ENVIRONMENTS
    ]
    evaluation_rows = [row for row in rows if row.split in EVALUATION_SPLITS]
    specialist_evaluation_rows = [
        row for row in evaluation_rows if row.environment in ELIGIBLE_ENVIRONMENTS
    ]
    indoor_evaluation_rows = [
        row for row in evaluation_rows if row.environment == "indoor"
    ]
    if not train_rows or not validation_rows or not specialist_evaluation_rows:
        raise ValueError(
            "train, validation, and specialist evaluation must all be non-empty"
        )
    if set(row.recording_id for row in train_rows) & set(
        row.recording_id for row in validation_rows + evaluation_rows
    ):
        raise ValueError("recording leakage found between training and held-out rows")
    development_groups = {
        row.source_group for row in [*train_rows, *validation_rows]
    }
    independent_evaluation_rows = [
        row
        for row in specialist_evaluation_rows
        if row.source_group not in development_groups
    ]
    repeated_group_diagnostic_rows = [
        row
        for row in specialist_evaluation_rows
        if row.source_group in development_groups
    ]
    if not independent_evaluation_rows:
        raise ValueError("source-group-independent evaluation must be non-empty")

    # Phase 1: model-family and regularization selection sees training rows only.
    candidates: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for l2 in L2_GRID:
            probabilities, cross_fit = grouped_cross_fit(train_rows, feature_set, l2)
            candidates.append(
                {
                    "featureSet": feature_set,
                    "l2": l2,
                    "outOfFold": {
                        **cross_fit,
                        "scoreMetricsAtSelectedThreshold": _metric_bundle(
                            train_rows,
                            probabilities,
                            probabilities
                            >= cross_fit["selectedThresholdMetrics"]["threshold"],
                        ),
                    },
                }
            )
    candidates.sort(key=_rank_candidate, reverse=True)
    selected = candidates[0]

    # Phase 2: refit the selected family, then select its operating point on validation.
    model = fit_specialist(
        train_rows, str(selected["featureSet"]), float(selected["l2"])
    )
    validation_probabilities = model.predict_proba(
        matrix_for(validation_rows, model.feature_set)
    )
    validation_threshold = select_threshold(
        labels_for(validation_rows), validation_probabilities
    )
    model = replace(model, threshold=float(validation_threshold["threshold"]))

    # Phase 3: the model and threshold are frozen before held-out evaluation is scored.
    specialist_probabilities = model.predict_proba(
        matrix_for(specialist_evaluation_rows, model.feature_set)
    )
    specialist_predictions = specialist_probabilities >= model.threshold
    independent_indices = np.asarray(
        [
            index
            for index, row in enumerate(specialist_evaluation_rows)
            if row.source_group not in development_groups
        ],
        dtype=np.int64,
    )
    policy_probabilities = np.asarray(
        [
            0.0
            if row.environment == "indoor"
            else model.predict_proba(matrix_for([row], model.feature_set))[0]
            for row in evaluation_rows
        ],
        dtype=np.float64,
    )
    policy_predictions = np.asarray(
        [
            False if row.environment == "indoor" else score >= model.threshold
            for row, score in zip(evaluation_rows, policy_probabilities, strict=True)
        ],
        dtype=bool,
    )

    created_at = datetime.now(UTC).isoformat()
    source_hashes = {
        "appearanceReport": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
            "kind": report.get("kind"),
            "createdAt": report.get("createdAt"),
        },
        "reviewDecisions": {
            "path": str(decision_path),
            "sha256": _sha256(decision_path),
            "savedAt": decisions.get("savedAt"),
        },
    }
    model_parameters = model.to_dict()
    fingerprint = model_fingerprint(model_parameters)
    model_payload = {
        "schemaVersion": MODEL_SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "createdAt": created_at,
        "fingerprint": fingerprint,
        "model": model_parameters,
        "selection": {
            "modelFamily": "maximum training recording-grouped OOF F1",
            "operatingThreshold": (
                "maximum validation F1; precision then recall tie-break"
            ),
            "l2Grid": list(L2_GRID),
            "selectedCandidate": selected,
        },
        "dataPolicy": {
            "train": "split=train and environment in {beach,grass}",
            "excludedFitRecordingIds": list(excluded_fit_recording_ids),
            "validation": "split=validation and non-indoor environment",
            "specialistEvaluation": (
                "split in {challenge,non-training,test} and non-indoor environment"
            ),
            "indoorPolicy": "always no-switch; indoor is outside specialist scope",
            "unclear": "excluded from fitting, threshold selection, and metrics",
        },
        "counts": {
            "review": review_counts,
            "eligibleTrainBeforeExclusion": _row_counts(eligible_train_rows),
            "train": _row_counts(train_rows),
            "validation": _row_counts(validation_rows),
            "specialistEvaluation": _row_counts(specialist_evaluation_rows),
            "independentEvaluation": _row_counts(independent_evaluation_rows),
            "repeatedGroupDiagnostic": _row_counts(repeated_group_diagnostic_rows),
            "indoorEvaluation": _row_counts(indoor_evaluation_rows),
        },
        "sources": source_hashes,
    }
    dataset_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-specialist-dataset-v1",
        "createdAt": created_at,
        "modelFingerprint": fingerprint,
        "featureSet": model.feature_set,
        "featureNames": list(model.feature_names),
        "reviewCounts": review_counts,
        "sources": source_hashes,
        "rows": [
            *(_dataset_row(row, model.feature_set, "train") for row in train_rows),
            *(
                _dataset_row(row, model.feature_set, "validation")
                for row in validation_rows
            ),
            *(
                _dataset_row(row, model.feature_set, "evaluation")
                for row in evaluation_rows
            ),
        ],
    }
    train_groups = {row.source_group for row in train_rows}
    evaluation_groups = {row.source_group for row in specialist_evaluation_rows}
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-specialist-evaluation-v1",
        "createdAt": created_at,
        "modelFingerprint": fingerprint,
        "modelPath": str(model_dir / "model.json"),
        "datasetPath": str(model_dir / "dataset.json"),
        "evaluationPath": str(evaluation_path),
        "threshold": model.threshold,
        "selectedFeatureSet": model.feature_set,
        "selectedL2": model.l2,
        "reviewCounts": review_counts,
        "counts": model_payload["counts"],
        "trainingSelectionLeaderboard": candidates,
        "validation": {
            "thresholdSelection": validation_threshold,
            "metrics": _metric_bundle(
                validation_rows,
                validation_probabilities,
                validation_probabilities >= model.threshold,
            ),
            "byRecording": _group_metrics(
                validation_rows,
                validation_probabilities,
                validation_probabilities >= model.threshold,
                lambda row: row.recording_id,
            ),
        },
        "heldOutSpecialist": _evaluation(
            specialist_evaluation_rows,
            specialist_probabilities,
            specialist_predictions,
        ),
        "heldOutIndependent": _evaluation(
            independent_evaluation_rows,
            specialist_probabilities[independent_indices],
            specialist_predictions[independent_indices],
        ),
        "heldOutProductPolicy": _evaluation(
            evaluation_rows, policy_probabilities, policy_predictions
        ),
        "knownLimitations": {
            "candidateConditioned": (
                "metrics apply only to generated markers, not missed real side switches"
            ),
            "sourceGroupOverlap": sorted(train_groups & evaluation_groups),
            "sourceGroupOverlapMeaning": (
                "recording IDs are disjoint, but these camera/source groups occur "
                "on both "
                "sides of the split; results are not source-group-independent"
            ),
            "indoorGate": (
                "indoor predictions are fixed to no-switch and are reported "
                "separately so "
                "they cannot inflate specialist precision/recall"
            ),
        },
        "predictions": [
            {
                "eventId": row.event_id,
                "recordingId": row.recording_id,
                "split": row.split,
                "environment": row.environment,
                "decision": row.decision,
                "score": float(score),
                "prediction": "switch" if prediction else "no-switch",
                "policy": (
                    "fixed-indoor-no-switch"
                    if row.environment == "indoor"
                    else "specialist"
                ),
            }
            for row, score, prediction in zip(
                evaluation_rows,
                policy_probabilities,
                policy_predictions,
                strict=True,
            )
        ],
        "sources": source_hashes,
    }

    _write_json(model_dir / "model.json", model_payload)
    _write_json(model_dir / "dataset.json", dataset_payload)
    _write_json(evaluation_path, evaluation_payload)
    return evaluation_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--exclude-fit-recording",
        action="append",
        default=[],
        help=(
            "recording ID to remove from fitting and grouped model-family selection; "
            "repeat for multiple IDs"
        ),
    )
    return parser


def main() -> None:
    result = train(_parser().parse_args())
    specialist = result["heldOutIndependent"]["overall"]
    policy = result["heldOutProductPolicy"]["overall"]
    print(f"model: {result['modelPath']}")
    print(f"evaluation: {result['evaluationPath']}")
    print(
        "selected: "
        f"{result['selectedFeatureSet']} l2={result['selectedL2']} "
        f"threshold={result['threshold']:.6f}"
    )
    print(
        "source-group-independent specialist: "
        f"precision={specialist['precision'] or 0:.4f} "
        f"recall={specialist['recall'] or 0:.4f} f1={specialist['f1'] or 0:.4f}"
    )
    print(
        "held-out product policy: "
        f"precision={policy['precision'] or 0:.4f} "
        f"recall={policy['recall'] or 0:.4f} f1={policy['f1'] or 0:.4f}"
    )


if __name__ == "__main__":
    main()
