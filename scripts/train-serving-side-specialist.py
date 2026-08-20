#!/usr/bin/env python3
"""Train and evaluate the first reviewed serving-side specialist.

Feature-family and regularization selection use recording-grouped out-of-fold
predictions from declared training recordings only. The operating threshold is
selected once on validation. Challenge, non-training, and protected-test labels are
used only after the model and threshold are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side_specialist import (
    EVALUATION_SPLITS,
    FEATURE_SETS,
    MODEL_KIND,
    MODEL_SCHEMA_VERSION,
    ReviewedRally,
    binary_metrics,
    fit_specialist,
    grouped_cross_fit,
    labels_for,
    matrix_for,
    model_fingerprint,
    rally_vector,
    reviewed_rallies,
    select_threshold,
)


DEFAULT_ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_REPORT = (
    DEFAULT_ROOT
    / "reports/serving-side/serving-side-existing-label-variants-full-nas-v2.json"
)
DEFAULT_DECISIONS = (
    DEFAULT_ROOT
    / "reports/serving-side/serving-side-review-decisions-full-nas-v1.json"
)
DEFAULT_MODEL_DIR = DEFAULT_ROOT / "models/serving-side-specialist-v1"
DEFAULT_EVALUATION = (
    DEFAULT_ROOT
    / "reports/serving-side/serving-side-specialist-v1-evaluation.json"
)
FROZEN_REPORT_SHA256 = (
    "611d698961534740b3b07624a2ade1d574de4e06b8bf5ce701b641b4690fc35b"
)
FROZEN_DECISIONS_SHA256 = (
    "1066edcf9579d3c92157a8504dbe5d798023ebb3ff4605e0dad9e451c97080b4"
)
L2_GRID = (0.01, 0.1, 1.0, 10.0)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_specialist.py",
    Path(__file__).resolve(),
)


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


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _row_counts(rows: Sequence[ReviewedRally]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "near": sum(row.label for row in rows),
        "far": sum(1 - row.label for row in rows),
        "recordings": len({row.recording_id for row in rows}),
        "recordingIds": sorted({row.recording_id for row in rows}),
        "sourceGroups": sorted({row.source_group for row in rows}),
    }


def _metric_bundle(
    rows: Sequence[ReviewedRally], probabilities: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    metrics = binary_metrics(labels_for(rows), predicted)
    labels = labels_for(rows).astype(np.int64)
    near_scores = probabilities[labels == 1]
    far_scores = probabilities[labels == 0]
    if len(near_scores) and len(far_scores):
        comparisons = near_scores[:, None] - far_scores[None, :]
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
        "rocAucNear": roc_auc,
        "averagePrecisionNear": average_precision,
        "meanNearProbability": (
            float(np.mean(near_scores)) if len(near_scores) else None
        ),
        "meanFarProbability": (
            float(np.mean(far_scores)) if len(far_scores) else None
        ),
    }


def _group_metrics(
    rows: Sequence[ReviewedRally],
    probabilities: np.ndarray,
    predicted: np.ndarray,
    key: Callable[[ReviewedRally], str],
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
    rows: Sequence[ReviewedRally], probabilities: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    return {
        "overall": _metric_bundle(rows, probabilities, predicted),
        "bySplit": _group_metrics(rows, probabilities, predicted, lambda row: row.split),
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
        float(metrics["balancedAccuracy"] or 0.0),
        float(metrics["macroF1"] or 0.0),
        float(metrics["accuracy"] or 0.0),
        -len(FEATURE_SETS[str(candidate["featureSet"])]),
        float(candidate["l2"]),
    )


def _dataset_row(
    row: ReviewedRally, feature_set: str, role: str
) -> dict[str, Any]:
    vector = rally_vector(row.rally, feature_set)
    return {
        "rallyId": row.rally_id,
        "recordingId": row.recording_id,
        "sourceGroup": row.source_group,
        "environment": row.environment,
        "sourceSplit": row.split,
        "role": role,
        "sourceType": row.source_type,
        "targetStatus": row.target_status,
        "serveAnchor": row.rally.get("start"),
        "decision": row.decision,
        "label": row.label,
        "features": [_finite_or_none(value) for value in vector],
    }


def _variant_score(row: ReviewedRally, variant: str) -> float:
    variants = row.rally.get("variants")
    evidence = variants.get(variant) if isinstance(variants, Mapping) else None
    score = evidence.get("score") if isinstance(evidence, Mapping) else None
    return float(score) if isinstance(score, (int, float)) and math.isfinite(score) else math.nan


def _heuristic_metrics(rows: Sequence[ReviewedRally], variant: str) -> dict[str, Any]:
    scores = np.asarray([_variant_score(row, variant) for row in rows], dtype=np.float64)
    usable = np.isfinite(scores)
    usable_rows = [row for row, keep in zip(rows, usable, strict=True) if keep]
    if not usable_rows:
        return {"rows": len(rows), "usableRows": 0, "coverage": 0.0}
    finite_scores = scores[usable]
    probabilities = np.clip((finite_scores + 1.0) / 2.0, 0.0, 1.0)
    return {
        "rows": len(rows),
        "usableRows": len(usable_rows),
        "coverage": len(usable_rows) / len(rows) if rows else None,
        "decisionRule": "signed score >= 0 predicts near; otherwise far",
        **_metric_bundle(usable_rows, probabilities, finite_scores >= 0.0),
    }


def _indices(
    rows: Sequence[ReviewedRally], predicate: Callable[[ReviewedRally], bool]
) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(rows) if predicate(row)],
        dtype=np.int64,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.serving_report.resolve()
    decision_path = args.decisions.resolve()
    model_dir = args.model_dir.resolve()
    evaluation_path = args.evaluation_output.resolve()
    if model_dir.exists():
        raise FileExistsError(f"refusing to reuse model directory: {model_dir}")
    if evaluation_path.exists():
        raise FileExistsError(
            f"refusing to overwrite evaluation artifact: {evaluation_path}"
        )
    report_hash = _sha256(report_path)
    decision_hash = _sha256(decision_path)
    if report_hash != FROZEN_REPORT_SHA256:
        raise ValueError("serving-side report does not match the frozen v1 SHA-256")
    if decision_hash != FROZEN_DECISIONS_SHA256:
        raise ValueError("serving-side decisions do not match the frozen v1 SHA-256")

    report = _load_json(report_path)
    decisions = _load_json(decision_path)
    rows, review_counts = reviewed_rallies(report, decisions)
    raw_decisions = decisions.get("decisions", {})
    if review_counts["missing"] or len(raw_decisions) != review_counts["rallies"]:
        raise ValueError(
            "training requires one saved review decision for every generated rally"
        )

    train_rows = [row for row in rows if row.split == "train"]
    validation_rows = [row for row in rows if row.split == "validation"]
    evaluation_rows = [row for row in rows if row.split in EVALUATION_SPLITS]
    if not train_rows or not validation_rows or not evaluation_rows:
        raise ValueError("train, validation, and evaluation must all be non-empty")
    train_recordings = {row.recording_id for row in train_rows}
    validation_recordings = {row.recording_id for row in validation_rows}
    evaluation_recordings = {row.recording_id for row in evaluation_rows}
    if (
        train_recordings & validation_recordings
        or train_recordings & evaluation_recordings
        or validation_recordings & evaluation_recordings
    ):
        raise ValueError("recording leakage found between declared data roles")
    development_groups = {
        row.source_group for row in [*train_rows, *validation_rows]
    }
    independent_evaluation_rows = [
        row for row in evaluation_rows if row.source_group not in development_groups
    ]
    raw_evaluation_rows = [
        row for row in evaluation_rows if row.split == "non-training"
    ]
    challenge_rows = [row for row in evaluation_rows if row.split == "challenge"]
    protected_test_rows = [row for row in evaluation_rows if row.split == "test"]
    if not independent_evaluation_rows or not raw_evaluation_rows or not protected_test_rows:
        raise ValueError(
            "independent, raw non-training, and protected-test scopes must be non-empty"
        )

    # Phase 1: family and regularization selection sees training rows only.
    candidates: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for l2 in L2_GRID:
            probabilities, cross_fit = grouped_cross_fit(train_rows, feature_set, l2)
            threshold = float(cross_fit["selectedThresholdMetrics"]["threshold"])
            candidates.append(
                {
                    "featureSet": feature_set,
                    "l2": l2,
                    "outOfFold": {
                        **cross_fit,
                        "scoreMetricsAtSelectedThreshold": _metric_bundle(
                            train_rows, probabilities, probabilities >= threshold
                        ),
                    },
                }
            )
    candidates.sort(key=_rank_candidate, reverse=True)
    selected = candidates[0]

    # Phase 2: refit the selected family and select one threshold on validation.
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

    # Phase 3: model and threshold are frozen before held-out labels are scored.
    evaluation_probabilities = model.predict_proba(
        matrix_for(evaluation_rows, model.feature_set)
    )
    evaluation_predictions = evaluation_probabilities >= model.threshold
    independent_indices = _indices(
        evaluation_rows, lambda row: row.source_group not in development_groups
    )
    raw_indices = _indices(evaluation_rows, lambda row: row.split == "non-training")
    challenge_indices = _indices(evaluation_rows, lambda row: row.split == "challenge")
    test_indices = _indices(evaluation_rows, lambda row: row.split == "test")

    created_at = datetime.now(UTC).isoformat()
    implementation = {
        "gitHeadBeforeArtifactCommit": _git_head(),
        "deterministic": True,
        "randomSeed": None,
        "files": [
            {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(path),
            }
            for path in IMPLEMENTATION_PATHS
        ],
    }
    sources = {
        "servingSideReport": {
            "path": str(report_path),
            "sha256": report_hash,
            "kind": report.get("kind"),
            "createdAt": report.get("createdAt"),
        },
        "reviewDecisions": {
            "path": str(decision_path),
            "sha256": decision_hash,
            "savedAt": decisions.get("savedAt"),
        },
        "implementation": implementation,
    }
    model_parameters = model.to_dict()
    fingerprint = model_fingerprint(model_parameters)
    counts = {
        "review": review_counts,
        "train": _row_counts(train_rows),
        "validation": _row_counts(validation_rows),
        "heldOutAll": _row_counts(evaluation_rows),
        "heldOutIndependent": _row_counts(independent_evaluation_rows),
        "heldOutRaw": _row_counts(raw_evaluation_rows),
        "heldOutChallenge": _row_counts(challenge_rows),
        "protectedTest": _row_counts(protected_test_rows),
    }
    model_payload = {
        "schemaVersion": MODEL_SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "createdAt": created_at,
        "fingerprint": fingerprint,
        "model": model_parameters,
        "selection": {
            "modelFamily": (
                "maximum training recording-grouped OOF balanced accuracy; "
                "macro-F1 then accuracy tie-break"
            ),
            "operatingThreshold": (
                "maximum validation balanced accuracy; macro-F1 then accuracy tie-break"
            ),
            "l2Grid": list(L2_GRID),
            "selectedCandidate": selected,
        },
        "dataPolicy": {
            "positiveDecision": "near",
            "negativeDecision": "far",
            "train": "split=train only",
            "validation": "split=validation only; threshold selection only",
            "heldOut": "split in {challenge,non-training,test}; evaluation only",
            "unclear": "excluded from fitting, selection, and metrics",
            "candidateConditioned": True,
        },
        "counts": counts,
        "sources": sources,
    }
    dataset_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-specialist-dataset-v1",
        "createdAt": created_at,
        "modelFingerprint": fingerprint,
        "featureSet": model.feature_set,
        "featureNames": list(model.feature_names),
        "reviewCounts": review_counts,
        "sources": sources,
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
    report_variants = report.get("variants")
    variant_names = sorted(report_variants) if isinstance(report_variants, Mapping) else []
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-specialist-evaluation-v1",
        "createdAt": created_at,
        "modelFingerprint": fingerprint,
        "modelPath": str(model_dir / "model.json"),
        "datasetPath": str(model_dir / "dataset.json"),
        "evaluationPath": str(evaluation_path),
        "threshold": model.threshold,
        "selectedFeatureSet": model.feature_set,
        "selectedL2": model.l2,
        "reviewCounts": review_counts,
        "counts": counts,
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
        "heldOutAll": _evaluation(
            evaluation_rows, evaluation_probabilities, evaluation_predictions
        ),
        "heldOutIndependent": _evaluation(
            independent_evaluation_rows,
            evaluation_probabilities[independent_indices],
            evaluation_predictions[independent_indices],
        ),
        "heldOutRaw": _evaluation(
            raw_evaluation_rows,
            evaluation_probabilities[raw_indices],
            evaluation_predictions[raw_indices],
        ),
        "heldOutChallenge": _evaluation(
            challenge_rows,
            evaluation_probabilities[challenge_indices],
            evaluation_predictions[challenge_indices],
        ),
        "protectedTest": _evaluation(
            protected_test_rows,
            evaluation_probabilities[test_indices],
            evaluation_predictions[test_indices],
        ),
        "heuristicBaselines": {
            variant: {
                "validation": _heuristic_metrics(validation_rows, variant),
                "heldOutRaw": _heuristic_metrics(raw_evaluation_rows, variant),
                "heldOutAll": _heuristic_metrics(evaluation_rows, variant),
            }
            for variant in variant_names
        },
        "knownLimitations": {
            "candidateConditioned": (
                "metrics classify reviewed generated rally rows; they do not measure "
                "missed rallies or serve-anchor recall"
            ),
            "sourceGroupOverlap": sorted(
                development_groups
                & {row.source_group for row in evaluation_rows}
            ),
            "rawScope": (
                "all raw non-training recordings share one source group, but their "
                "recording IDs and labels were not used for model or threshold selection"
            ),
            "protectedTest": (
                "the test split is reported once after training and validation selection"
            ),
            "existingFeatureOnly": (
                "v1 uses the already-extracted whole-half, baseline-band, and HOG "
                "change features; it adds no court calibration or player tracking"
            ),
        },
        "predictions": [
            {
                "rallyId": row.rally_id,
                "recordingId": row.recording_id,
                "split": row.split,
                "environment": row.environment,
                "decision": row.decision,
                "nearProbability": float(probability),
                "prediction": "near" if prediction else "far",
            }
            for row, probability, prediction in zip(
                evaluation_rows,
                evaluation_probabilities,
                evaluation_predictions,
                strict=True,
            )
        ],
        "sources": sources,
    }

    _write_json(model_dir / "model.json", model_payload)
    _write_json(model_dir / "dataset.json", dataset_payload)
    _write_json(evaluation_path, evaluation_payload)
    return evaluation_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVALUATION)
    return parser


def main() -> None:
    result = train(_parser().parse_args())
    raw = result["heldOutRaw"]["overall"]
    independent = result["heldOutIndependent"]["overall"]
    print(f"model: {result['modelPath']}")
    print(f"evaluation: {result['evaluationPath']}")
    print(
        "selected: "
        f"{result['selectedFeatureSet']} l2={result['selectedL2']} "
        f"threshold={result['threshold']:.6f}"
    )
    print(
        "raw non-training: "
        f"balanced_accuracy={raw['balancedAccuracy'] or 0:.4f} "
        f"macro_f1={raw['macroF1'] or 0:.4f} accuracy={raw['accuracy'] or 0:.4f}"
    )
    print(
        "source-group-independent: "
        f"balanced_accuracy={independent['balancedAccuracy'] or 0:.4f} "
        f"macro_f1={independent['macroF1'] or 0:.4f} "
        f"accuracy={independent['accuracy'] or 0:.4f}"
    )


if __name__ == "__main__":
    main()
