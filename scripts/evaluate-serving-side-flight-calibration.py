#!/usr/bin/env python3
"""Calibrate fixed-flight probabilities and freeze development abstention bands."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side_abstention import (
    select_precision_operating_point,
    selective_metrics,
)
from analysis.serving_side_calibration import calibration_metrics, fit_platt


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_EVALUATION = (
    ROOT / "reports/serving-side/serving-side-flight-v3-development.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/serving-side/serving-side-flight-v3-calibration-abstention-development.json"
)
PLATT_L2_GRID = (0.01, 0.1, 1.0, 10.0)
PRECISION_TARGETS = (0.94, 0.95, 0.96, 0.97, 0.98)
RECOMMENDED_PRECISION_TARGET = 0.95
MINIMUM_PREDICTIONS_PER_SIDE = 50
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_abstention.py",
    REPOSITORY_ROOT / "analysis/serving_side_calibration.py",
    Path(__file__).resolve(),
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _by_group(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray
) -> dict[str, Any]:
    result = {}
    for group in sorted({str(row["sourceGroup"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == group]
        )
        result[group] = calibration_metrics(
            [int(rows[index]["label"]) for index in indices], probabilities[indices]
        )
    return result


def _nested_platt(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    truth: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    calibrated = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    for group in sorted({str(row["sourceGroup"]) for row in rows}):
        held = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == group]
        )
        train = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] != group]
        )
        model = fit_platt(probabilities[train], truth[train], l2=l2)
        if model.coefficient <= 0:
            raise ValueError(f"non-monotonic Platt fold for {group} at L2 {l2}")
        calibrated[held] = model.predict(probabilities[held])
        folds.append(
            {
                "heldSourceGroup": group,
                "trainRows": len(train),
                "heldRows": len(held),
                "parameters": model.to_dict(),
            }
        )
    if not np.isfinite(calibrated).all():
        raise AssertionError("nested calibration left development rows unscored")
    return calibrated, folds


def _calibration_candidate(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
    probabilities: np.ndarray,
    folds: Sequence[Mapping[str, Any]],
    complexity: int,
) -> dict[str, Any]:
    by_group = _by_group(rows, probabilities)
    return {
        "name": name,
        "complexity": complexity,
        "metrics": calibration_metrics(truth, probabilities),
        "worstSourceGroupBrier": max(
            float(metrics["brier"]) for metrics in by_group.values()
        ),
        "bySourceGroup": by_group,
        "folds": list(folds),
        "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
    }


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    return (
        float(metrics["brier"]),
        float(metrics["logLoss"]),
        float(candidate["worstSourceGroupBrier"]),
        int(candidate["complexity"]),
    )


def _with_review_fraction(point: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if point is None:
        return None
    return {**point, "reviewFraction": 1.0 - float(point["coverage"])}


def _selective_by_group(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    point: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for group in sorted({str(row["sourceGroup"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == group]
        )
        result[group] = _with_review_fraction(
            selective_metrics(
                [int(rows[index]["label"]) for index in indices],
                probabilities[indices],
                far_threshold=float(point["farThreshold"]),
                near_threshold=float(point["nearThreshold"]),
            )
        )
    return result


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    evaluation_path = args.evaluation.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration report: {output_path}")
    evaluation = _load(evaluation_path)
    if evaluation.get("kind") != "volleycut-serving-side-flight-development-evaluation-v1":
        raise ValueError("unexpected fixed-flight evaluation kind")
    selected = evaluation.get("selection", {}).get("selectedCandidate")
    raw_predictions = evaluation.get("selectedPredictions")
    source = evaluation.get("sources", {}).get("flightDevelopmentDataset")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(raw_predictions, list)
        or not raw_predictions
        or not isinstance(source, Mapping)
    ):
        raise ValueError("fixed-flight selection inputs are unavailable")
    dataset_path = Path(str(source["path"])).resolve()
    if _sha256(dataset_path) != source.get("sha256"):
        raise ValueError("fixed-flight development dataset hash changed")
    dataset = _load(dataset_path)
    if (
        dataset.get("scope") != "development"
        or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False
    ):
        raise ValueError("calibration requires correction-clean development data")
    rows_value = dataset.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("development rows are unavailable")
    rows = [row for row in rows_value if isinstance(row, Mapping)]
    if len(rows) != len(rows_value) or any(row.get("sourceSplit") == "test" for row in rows):
        raise ValueError("calibration rows are malformed or protected")
    prediction_by_id = {
        str(row["rallyId"]): row for row in raw_predictions if isinstance(row, Mapping)
    }
    if len(prediction_by_id) != len(rows):
        raise ValueError("selected predictions do not cover development rows exactly")
    probabilities = np.asarray(
        [float(prediction_by_id[str(row["rallyId"])]["probabilityNear"]) for row in rows]
    )
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if any(
        prediction_by_id[str(row["rallyId"])].get("decision") != row.get("decision")
        for row in rows
    ):
        raise ValueError("evaluation decisions and correction-clean labels differ")
    expected_digest = selected.get("evaluation", {}).get("predictionDigest")
    if hashlib.sha256(probabilities.tobytes()).hexdigest() != expected_digest:
        raise ValueError("fixed-flight prediction digest does not reproduce")
    center_threshold = float(
        selected["evaluation"]["thresholdSelection"]["threshold"]
    )

    calibrated_probabilities: dict[str, np.ndarray] = {"identity": probabilities}
    candidates = [
        _calibration_candidate("identity", rows, truth, probabilities, [], 0)
    ]
    for l2 in PLATT_L2_GRID:
        calibrated, folds = _nested_platt(rows, probabilities, truth, l2)
        name = f"platt-l2-{l2:g}"
        calibrated_probabilities[name] = calibrated
        candidates.append(
            _calibration_candidate(name, rows, truth, calibrated, folds, 1)
        )
    selected_calibration = min(candidates, key=_candidate_rank)
    selected_name = str(selected_calibration["name"])
    if selected_name == "identity":
        deployable_calibrator = None
    else:
        selected_l2 = float(selected_name.rsplit("-", 1)[1])
        deployable_calibrator = fit_platt(probabilities, truth, l2=selected_l2)
        if deployable_calibrator.coefficient <= 0:
            raise ValueError("deployable Platt calibration is non-monotonic")

    baseline = _with_review_fraction(
        selective_metrics(
            truth,
            probabilities,
            far_threshold=center_threshold,
            near_threshold=center_threshold,
        )
    )
    target_points = {
        f"{target:.2f}": _with_review_fraction(
            select_precision_operating_point(
                truth,
                probabilities,
                center_threshold=center_threshold,
                precision_floor=target,
                minimum_predictions_per_side=MINIMUM_PREDICTIONS_PER_SIDE,
            )
        )
        for target in PRECISION_TARGETS
    }
    recommended = target_points[f"{RECOMMENDED_PRECISION_TARGET:.2f}"]
    if recommended is None:
        raise ValueError("predeclared 95% precision operating point is infeasible")
    full_calibrated = (
        probabilities
        if deployable_calibrator is None
        else deployable_calibrator.predict(probabilities)
    )
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-flight-v3-calibration-abstention-development-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": "development-nested-cross-fit",
        "modelFingerprint": evaluation["finalModel"]["fingerprint"],
        "featureFamily": selected["featureFamily"],
        "centerThreshold": center_threshold,
        "calibrationSelection": {
            "primaryMetric": "lower pooled Brier score",
            "tieBreaks": [
                "lower pooled log loss",
                "lower worst-source-group Brier score",
                "simpler candidate",
            ],
            "method": "nested leave-one-source-group-out over fixed model cross-fit probabilities",
            "selectedCandidate": selected_name,
            "leaderboard": sorted(candidates, key=_candidate_rank),
        },
        "deployableCalibration": {
            "family": "identity" if deployable_calibrator is None else "platt",
            "parameters": (
                None if deployable_calibrator is None else deployable_calibrator.to_dict()
            ),
            "trainingInput": "all development leave-one-source-group-out fixed-model probabilities",
            "metricsOnTrainingInput": calibration_metrics(truth, full_calibrated),
        },
        "abstentionSelection": {
            "score": "raw fixed-flight near probability",
            "minimumPredictionsPerSide": MINIMUM_PREDICTIONS_PER_SIDE,
            "precisionTargets": list(PRECISION_TARGETS),
            "recommendedPrecisionTarget": RECOMMENDED_PRECISION_TARGET,
            "baseline": baseline,
            "precisionTargetOperatingPoints": target_points,
            "recommended": recommended,
            "recommendedBySourceGroup": _selective_by_group(
                rows, probabilities, recommended
            ),
            "policy": "existing side decisions may only become abstentions",
        },
        "predictions": [
            {
                "rallyId": row["rallyId"],
                "recordingId": row["recordingId"],
                "sourceGroup": row["sourceGroup"],
                "decision": row["decision"],
                "rawNearProbability": float(raw),
                "nestedCalibratedNearProbability": float(calibrated),
                "recommendedDecision": (
                    "far"
                    if raw < float(recommended["farThreshold"])
                    else "near"
                    if raw >= float(recommended["nearThreshold"])
                    else "review"
                ),
            }
            for row, raw, calibrated in zip(
                rows,
                probabilities,
                calibrated_probabilities[selected_name],
                strict=True,
            )
        ],
        "selectionPolicy": {
            "calibrationCandidates": ["identity", *[f"platt L2 {value:g}" for value in PLATT_L2_GRID]],
            "precisionTargets": list(PRECISION_TARGETS),
            "recommendedTargetFrozenBeforeEvaluation": RECOMMENDED_PRECISION_TARGET,
            "thresholdConstraint": "far <= frozen center <= near; no side flips",
        },
        "sources": {
            "fixedFlightEvaluation": {
                "path": str(evaluation_path),
                "sha256": _sha256(evaluation_path),
            },
            "developmentDataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
            },
            "fixedPredictionDigest": expected_digest,
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ],
        },
        "dataPolicy": {
            "humanCorrectionsApplied": True,
            "sourceQualityExclusionsApplied": True,
            "protectedTestLoaded": False,
            "protectedTestUsedForSelection": False,
        },
    }
    atomic_write_text(output_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output_path}")
    print(
        json.dumps(
            {
                "selectedCalibration": selected_name,
                "baseline": baseline,
                "recommended": recommended,
            },
            indent=2,
        )
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
