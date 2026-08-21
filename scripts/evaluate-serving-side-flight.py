#!/usr/bin/env python3
"""Evaluate concentrated flight-motion resolution and grid ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side_specialist import binary_metrics, select_threshold
from analysis.serving_side_v2 import fit_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_DATASET = ROOT / "features/serving-side-flight-v1/development.json"
DEFAULT_OUTPUT = (
    ROOT / "reports/serving-side/serving-side-flight-v1-development.json"
)
L2_GRID = (0.01, 0.1, 1.0, 10.0)
SCREENING_L2 = 0.1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_flight.py",
    REPOSITORY_ROOT / "scripts/extract-serving-side-flight-features.py",
    Path(__file__).resolve(),
)


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def _raw_flight_matrix(
    rows: Sequence[Mapping[str, Any]],
    configuration: str,
    feature_names: Sequence[str],
) -> np.ndarray:
    values = []
    for row in rows:
        configurations = row.get("configurations")
        features = (
            configurations.get(configuration)
            if isinstance(configurations, Mapping)
            else None
        )
        if not isinstance(features, Mapping):
            raise ValueError(
                f"row {row.get('rallyId')} lacks configuration {configuration}"
            )
        values.append([float(features[name]) for name in feature_names])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"configuration {configuration} has non-finite features")
    return result


def _v2_matrix(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    values = []
    for row in rows:
        features = row.get("v2RecordingRankFeatures")
        if not isinstance(features, Mapping):
            raise ValueError(f"row {row.get('rallyId')} lacks v2 rank features")
        values.append([float(features[name]) for name in feature_names])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("v2 rank features contain non-finite values")
    return result


def _tied_recording_ranks(
    rows: Sequence[Mapping[str, Any]], values: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["recordingId"] == recording_id
            ]
        )
        count = len(indices)
        local = values[indices]
        for feature in range(values.shape[1]):
            order = np.argsort(local[:, feature], kind="stable")
            sorted_values = local[order, feature]
            ranks = np.zeros(count, dtype=np.float64)
            start = 0
            while start < count:
                end = start + 1
                while end < count and sorted_values[end] == sorted_values[start]:
                    end += 1
                rank = (
                    0.5
                    if count == 1
                    else ((start + end - 1) / 2.0) / (count - 1)
                )
                ranks[order[start:end]] = rank
                start = end
            result[indices, feature] = ranks
    return result


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, key: str
) -> dict[str, Any]:
    truth = _labels(rows)
    output: dict[str, Any] = {}
    for group in sorted({str(row[key]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row[key] == group]
        )
        output[group] = binary_metrics(truth[indices], predicted[indices])
    return output


def _cross_fit(
    rows: Sequence[Mapping[str, Any]], values: np.ndarray, l2: float
) -> tuple[dict[str, Any], np.ndarray]:
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("flight evaluation needs at least three source groups")
    truth = _labels(rows)
    probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    for group in groups:
        held = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["sourceGroup"] == group
            ]
        )
        train = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["sourceGroup"] != group
            ]
        )
        model = fit_logistic(values[train], truth[train], l2=l2)
        probabilities[held] = model.predict_proba(values[held])
        folds.append(
            {
                "heldSourceGroup": group,
                "trainRows": len(train),
                "heldRows": len(held),
            }
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("source-group cross-fit left rows unscored")
    threshold_metrics = select_threshold(truth, probabilities)
    threshold = float(threshold_metrics["threshold"])
    predicted = probabilities >= threshold
    by_group = _group_metrics(rows, predicted, "sourceGroup")
    balanced = [
        float(metrics["balancedAccuracy"])
        for metrics in by_group.values()
        if metrics["balancedAccuracy"] is not None
    ]
    audit = {
        "thresholdSelection": threshold_metrics,
        "pooledMetrics": binary_metrics(truth, predicted),
        "sourceGroupMacroBalancedAccuracy": float(np.mean(balanced)),
        "worstSourceGroupBalancedAccuracy": float(np.min(balanced)),
        "bySourceGroup": by_group,
        "byEnvironment": _group_metrics(rows, predicted, "environment"),
        "folds": folds,
        "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
    }
    return audit, probabilities


def _rank(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = candidate["evaluation"]["pooledMetrics"]
    return (
        float(candidate["evaluation"]["sourceGroupMacroBalancedAccuracy"]),
        float(metrics["balancedAccuracy"] or 0.0),
        float(metrics["macroF1"] or 0.0),
        float(candidate["evaluation"]["worstSourceGroupBalancedAccuracy"]),
        -int(candidate["featureCount"]),
    )


def _candidate(
    rows: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    names: Sequence[str],
    *,
    stage: str,
    configuration: str | None,
    feature_family: str,
    l2: float,
) -> tuple[dict[str, Any], np.ndarray]:
    audit, probabilities = _cross_fit(rows, values, l2)
    return (
        {
            "stage": stage,
            "configuration": configuration,
            "featureFamily": feature_family,
            "featureCount": len(names),
            "l2": l2,
            "evaluation": audit,
        },
        probabilities,
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite flight evaluation: {output_path}")
    dataset = _load(dataset_path)
    if (
        dataset.get("kind")
        != "volleycut-serving-side-flight-feature-development-v1"
        or dataset.get("scope") != "development"
        or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or not dataset.get("dataPolicy", {}).get("protectedSourceGroupsExcluded")
        or dataset.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("flight selection requires the complete development artifact")
    raw_rows = dataset.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("flight dataset rows are unavailable")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows) or any(row.get("sourceSplit") == "test" for row in rows):
        raise ValueError("flight dataset contains invalid or protected rows")
    configurations = dataset.get("configurations")
    v2_names = dataset.get("v2FeatureNames")
    if (
        not isinstance(configurations, list)
        or not configurations
        or not isinstance(v2_names, list)
        or not v2_names
    ):
        raise ValueError("flight feature contracts are unavailable")
    v2_names = [str(name) for name in v2_names]
    v2_values = _v2_matrix(rows, v2_names)
    matrices: dict[str, dict[str, Any]] = {}
    for configuration in configurations:
        if not isinstance(configuration, Mapping):
            raise ValueError("flight configuration is invalid")
        name = str(configuration["name"])
        names = configuration.get("featureNames")
        if not isinstance(names, list) or not names:
            raise ValueError(f"configuration {name} has no feature signature")
        raw = _raw_flight_matrix(rows, name, names)
        matrices[name] = {
            "names": [str(item) for item in names],
            "raw": raw,
            "rank": _tied_recording_ranks(rows, raw),
            "resize": configuration["resize"],
            "grid": configuration["grid"],
        }

    screening: list[dict[str, Any]] = []
    for name, configuration in matrices.items():
        for family, values, names in (
            (
                "flight-recording-rank",
                configuration["rank"],
                [f"flight:{item}" for item in configuration["names"]],
            ),
            (
                "v2-plus-flight-recording-rank",
                np.column_stack((v2_values, configuration["rank"])),
                [*(f"v2:{item}" for item in v2_names), *(f"flight:{item}" for item in configuration["names"])],
            ),
        ):
            print(f"screening {name} / {family}", flush=True)
            candidate, _ = _candidate(
                rows,
                values,
                names,
                stage="resolution-grid-screening",
                configuration=name,
                feature_family=family,
                l2=SCREENING_L2,
            )
            candidate["resize"] = configuration["resize"]
            candidate["grid"] = configuration["grid"]
            screening.append(candidate)
    screening.sort(key=_rank, reverse=True)
    selected_configuration = str(screening[0]["configuration"])
    selected_matrix = matrices[selected_configuration]

    families = {
        "v2-recording-rank": (
            v2_values,
            [f"v2:{item}" for item in v2_names],
        ),
        "flight-absolute": (
            selected_matrix["raw"],
            [f"flight:{item}" for item in selected_matrix["names"]],
        ),
        "flight-recording-rank": (
            selected_matrix["rank"],
            [f"flight:{item}" for item in selected_matrix["names"]],
        ),
        "v2-plus-flight-recording-rank": (
            np.column_stack((v2_values, selected_matrix["rank"])),
            [
                *(f"v2:{item}" for item in v2_names),
                *(f"flight:{item}" for item in selected_matrix["names"]),
            ],
        ),
    }
    tuning: list[dict[str, Any]] = []
    probabilities_by_key: dict[tuple[str, float], np.ndarray] = {}
    for family, (values, names) in families.items():
        for l2 in L2_GRID:
            print(
                f"tuning {selected_configuration} / {family} / l2={l2}",
                flush=True,
            )
            candidate, probabilities = _candidate(
                rows,
                values,
                names,
                stage="selected-configuration-tuning",
                configuration=(
                    None if family == "v2-recording-rank" else selected_configuration
                ),
                feature_family=family,
                l2=l2,
            )
            tuning.append(candidate)
            probabilities_by_key[(family, l2)] = probabilities
    tuning.sort(key=_rank, reverse=True)
    selected = tuning[0]
    selected_key = (str(selected["featureFamily"]), float(selected["l2"]))
    selected_probabilities = probabilities_by_key[selected_key]
    selected_threshold = float(
        selected["evaluation"]["thresholdSelection"]["threshold"]
    )
    selected_predictions = selected_probabilities >= selected_threshold
    selected_values, selected_names = families[selected_key[0]]
    final_model = fit_logistic(
        selected_values, _labels(rows), l2=selected_key[1]
    )
    final_model = replace(final_model, threshold=selected_threshold)
    final_parameters = final_model.to_dict()
    final_fingerprint = hashlib.sha256(
        json.dumps(
            final_parameters, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    top_weights = sorted(
        (
            {
                "feature": name,
                "weight": float(weight),
                "absoluteWeight": abs(float(weight)),
            }
            for name, weight in zip(
                selected_names, final_model.weights, strict=True
            )
        ),
        key=lambda item: (-item["absoluteWeight"], item["feature"]),
    )[:30]
    baseline = max(
        (
            candidate
            for candidate in tuning
            if candidate["featureFamily"] == "v2-recording-rank"
        ),
        key=_rank,
    )
    baseline_key = ("v2-recording-rank", float(baseline["l2"]))
    baseline_probabilities = probabilities_by_key[baseline_key]
    baseline_threshold = float(
        baseline["evaluation"]["thresholdSelection"]["threshold"]
    )
    baseline_predictions = baseline_probabilities >= baseline_threshold
    truth = _labels(rows)
    paired = {
        "bothCorrect": int(np.sum((selected_predictions == truth) & (baseline_predictions == truth))),
        "selectedOnlyCorrect": int(np.sum((selected_predictions == truth) & (baseline_predictions != truth))),
        "baselineOnlyCorrect": int(np.sum((selected_predictions != truth) & (baseline_predictions == truth))),
        "bothWrong": int(np.sum((selected_predictions != truth) & (baseline_predictions != truth))),
    }
    created = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-flight-development-evaluation-v1",
        "createdAt": created,
        "selection": {
            "primaryMetric": "mean source-group balanced accuracy",
            "tieBreaks": [
                "pooled balanced accuracy",
                "pooled macro-F1",
                "worst source-group balanced accuracy",
                "fewer features",
            ],
            "crossValidation": "leave-one-source-group-out over correction-clean development rows",
            "protectedTest": "not loaded, scored, or used",
            "screeningL2": SCREENING_L2,
            "selectedConfiguration": selected_configuration,
            "selectedCandidate": selected,
            "baselineCandidate": baseline,
        },
        "screeningLeaderboard": screening,
        "tuningLeaderboard": tuning,
        "pairedAgainstCorrectionCleanV2": paired,
        "selectedTopWeights": top_weights,
        "finalModel": {
            "fingerprint": final_fingerprint,
            "configuration": selected_configuration,
            "featureFamily": selected_key[0],
            "featureNames": selected_names,
            "parameters": final_parameters,
            "trainingRows": len(rows),
        },
        "selectedPredictions": [
            {
                "rallyId": row["rallyId"],
                "recordingId": row["recordingId"],
                "sourceGroup": row["sourceGroup"],
                "environment": row["environment"],
                "decision": row["decision"],
                "probabilityNear": float(probability),
                "prediction": "near" if prediction else "far",
                "correct": bool(prediction == label),
            }
            for row, probability, prediction, label in zip(
                rows,
                selected_probabilities,
                selected_predictions,
                truth,
                strict=True,
            )
        ],
        "counts": dataset.get("counts"),
        "sources": {
            "flightDevelopmentDataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
            },
            "humanLabelCorrections": dataset.get("sources", {}).get(
                "humanLabelCorrections"
            ),
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ],
        },
    }
    atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(f"selected {selected_key[0]} / {selected_configuration}")
    print(f"wrote {output_path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
