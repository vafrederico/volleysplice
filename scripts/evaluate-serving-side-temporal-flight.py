#!/usr/bin/env python3
"""Evaluate inference-safe early/center/late flight windows on development data."""

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
from analysis.serving_side_review_slices import rescore_reviewed_predictions
from analysis.serving_side_specialist import binary_metrics, select_threshold
from analysis.serving_side_v2 import fit_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_DATASET = ROOT / "features/serving-side-temporal-flight-v1/development.json"
DEFAULT_REVIEW_SLICES = (
    ROOT / "reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/serving-side/serving-side-temporal-flight-v2-development.json"
)
L2_GRID = (0.01, 0.1, 1.0, 10.0)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_review_slices.py",
    REPOSITORY_ROOT / "analysis/serving_side_v2.py",
    REPOSITORY_ROOT / "scripts/extract-serving-side-temporal-flight-features.py",
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


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def _matrix(
    rows: Sequence[Mapping[str, Any]], field: str, names: Sequence[str]
) -> np.ndarray:
    result = np.asarray(
        [[float(row[field][name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise ValueError(f"{field} contains non-finite features")
    return result


def _tied_recording_ranks(
    rows: Sequence[Mapping[str, Any]], values: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["recordingId"] == recording_id]
        )
        local = values[indices]
        count = len(indices)
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
    return {
        group: binary_metrics(
            truth[
                np.asarray([index for index, row in enumerate(rows) if row[key] == group])
            ],
            predicted[
                np.asarray([index for index, row in enumerate(rows) if row[key] == group])
            ],
        )
        for group in sorted({str(row[key]) for row in rows})
    }


def _cross_fit(
    rows: Sequence[Mapping[str, Any]], values: np.ndarray, l2: float
) -> tuple[dict[str, Any], np.ndarray]:
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("temporal evaluation requires at least three source groups")
    truth = _labels(rows)
    probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    for group in groups:
        held = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == group]
        )
        train = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] != group]
        )
        model = fit_logistic(values[train], truth[train], l2=l2)
        probabilities[held] = model.predict_proba(values[held])
        folds.append(
            {"heldSourceGroup": group, "trainRows": len(train), "heldRows": len(held)}
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("cross-fit left rows unscored")
    threshold = select_threshold(truth, probabilities)
    predicted = probabilities >= float(threshold["threshold"])
    by_group = _group_metrics(rows, predicted, "sourceGroup")
    balanced = [
        float(metrics["balancedAccuracy"])
        for metrics in by_group.values()
        if metrics["balancedAccuracy"] is not None
    ]
    return (
        {
            "thresholdSelection": threshold,
            "pooledMetrics": binary_metrics(truth, predicted),
            "sourceGroupMacroBalancedAccuracy": float(np.mean(balanced)),
            "worstSourceGroupBalancedAccuracy": float(np.min(balanced)),
            "bySourceGroup": by_group,
            "byEnvironment": _group_metrics(rows, predicted, "environment"),
            "folds": folds,
            "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
        },
        probabilities,
    )


def _rank(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    evaluation = candidate["evaluation"]
    pooled = evaluation["pooledMetrics"]
    return (
        float(evaluation["sourceGroupMacroBalancedAccuracy"]),
        float(pooled["balancedAccuracy"] or 0),
        float(pooled["macroF1"] or 0),
        float(evaluation["worstSourceGroupBalancedAccuracy"]),
        -int(candidate["featureCount"]),
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset_path = args.dataset.resolve()
    review_path = args.review_slices.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite temporal evaluation: {output_path}")
    dataset = _load(dataset_path)
    if (
        dataset.get("kind")
        != "volleycut-serving-side-temporal-flight-feature-development-v1"
        or dataset.get("scope") != "development"
        or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or dataset.get("dataPolicy", {}).get("humanCorrectedAnchorsUsed") is not False
    ):
        raise ValueError("temporal selection requires an inference-safe development artifact")
    temporal_raw = dataset.get("rows")
    if not isinstance(temporal_raw, list) or not temporal_raw:
        raise ValueError("temporal dataset has no rows")
    temporal_rows = [row for row in temporal_raw if isinstance(row, Mapping)]
    if len(temporal_rows) != len(temporal_raw):
        raise ValueError("temporal dataset contains invalid rows")

    center_source = dataset.get("sources", {}).get("centerFlightDataset")
    if not isinstance(center_source, Mapping):
        raise ValueError("temporal dataset lacks its center source")
    center_path = Path(str(center_source["path"])).resolve()
    if _sha256(center_path) != center_source.get("sha256"):
        raise ValueError("center dataset changed since temporal extraction")
    center = _load(center_path)
    center_by_id = {
        str(row["rallyId"]): row
        for row in center.get("rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("rallyId"), str)
    }
    rows = []
    for temporal in temporal_rows:
        rally_id = str(temporal["rallyId"])
        center_row = center_by_id.get(rally_id)
        if center_row is None:
            raise ValueError(f"center dataset lacks {rally_id}")
        if any(
            temporal[field] != center_row[field]
            for field in ("recordingId", "sourceGroup", "environment", "label")
        ):
            raise ValueError(f"temporal and center identities differ for {rally_id}")
        rows.append({**center_row, "shiftedFlightFeatures": temporal["shiftedFlightFeatures"]})

    configuration = dataset.get("configuration")
    feature_names = configuration.get("featureNames") if isinstance(configuration, Mapping) else None
    v2_names = center.get("v2FeatureNames")
    if not isinstance(feature_names, list) or not isinstance(v2_names, list):
        raise ValueError("temporal feature contracts are unavailable")
    feature_names = [str(name) for name in feature_names]
    v2_names = [str(name) for name in v2_names]
    v2 = _matrix(rows, "v2RecordingRankFeatures", v2_names)
    center_raw = np.asarray(
        [[float(row["configurations"]["192x108-r4c6"][name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    minus_raw = np.asarray(
        [[float(row["shiftedFlightFeatures"]["minus1"][name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    plus_raw = np.asarray(
        [[float(row["shiftedFlightFeatures"]["plus1"][name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    if not all(np.isfinite(values).all() for values in (center_raw, minus_raw, plus_raw)):
        raise ValueError("temporal flight features contain non-finite values")
    minus_rank, center_rank, plus_rank = (
        _tied_recording_ranks(rows, values)
        for values in (minus_raw, center_raw, plus_raw)
    )
    mean_rank = _tied_recording_ranks(
        rows, np.mean(np.stack((minus_raw, center_raw, plus_raw)), axis=0)
    )
    range_rank = _tied_recording_ranks(
        rows, np.ptp(np.stack((minus_raw, center_raw, plus_raw)), axis=0)
    )
    v2_labels = [f"v2:{name}" for name in v2_names]
    flight_labels = [f"flight:{name}" for name in feature_names]
    families = {
        "v2-plus-center-flight-rank": (
            np.column_stack((v2, center_rank)),
            [*v2_labels, *(f"center:{name}" for name in flight_labels)],
        ),
        "v2-plus-minus1-flight-rank": (
            np.column_stack((v2, minus_rank)),
            [*v2_labels, *(f"minus1:{name}" for name in flight_labels)],
        ),
        "v2-plus-plus1-flight-rank": (
            np.column_stack((v2, plus_rank)),
            [*v2_labels, *(f"plus1:{name}" for name in flight_labels)],
        ),
        "temporal-flight-rank-concat": (
            np.column_stack((minus_rank, center_rank, plus_rank)),
            [
                *(f"minus1:{name}" for name in flight_labels),
                *(f"center:{name}" for name in flight_labels),
                *(f"plus1:{name}" for name in flight_labels),
            ],
        ),
        "v2-plus-temporal-flight-rank-concat": (
            np.column_stack((v2, minus_rank, center_rank, plus_rank)),
            [
                *v2_labels,
                *(f"minus1:{name}" for name in flight_labels),
                *(f"center:{name}" for name in flight_labels),
                *(f"plus1:{name}" for name in flight_labels),
            ],
        ),
        "v2-plus-temporal-mean-range-rank": (
            np.column_stack((v2, mean_rank, range_rank)),
            [
                *v2_labels,
                *(f"temporalMean:{name}" for name in flight_labels),
                *(f"temporalRange:{name}" for name in flight_labels),
            ],
        ),
    }

    leaderboard: list[dict[str, Any]] = []
    probabilities_by_key: dict[tuple[str, float], np.ndarray] = {}
    for family, (values, names) in families.items():
        for l2 in L2_GRID:
            print(f"evaluating {family} / l2={l2}", flush=True)
            audit, probabilities = _cross_fit(rows, values, l2)
            leaderboard.append(
                {
                    "featureFamily": family,
                    "featureCount": len(names),
                    "l2": l2,
                    "evaluation": audit,
                }
            )
            probabilities_by_key[(family, l2)] = probabilities
    leaderboard.sort(key=_rank, reverse=True)
    selected = leaderboard[0]
    selected_key = (str(selected["featureFamily"]), float(selected["l2"]))
    best_temporal = next(
        candidate
        for candidate in leaderboard
        if "temporal" in str(candidate["featureFamily"])
    )
    best_temporal_key = (
        str(best_temporal["featureFamily"]),
        float(best_temporal["l2"]),
    )
    selected_values, selected_names = families[selected_key[0]]
    selected_probabilities = probabilities_by_key[selected_key]
    threshold = float(selected["evaluation"]["thresholdSelection"]["threshold"])
    predicted = selected_probabilities >= threshold
    truth = _labels(rows)
    final_model = replace(
        fit_logistic(selected_values, truth, l2=selected_key[1]), threshold=threshold
    )
    final_parameters = final_model.to_dict()
    fingerprint = hashlib.sha256(
        json.dumps(final_parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    selected_predictions = [
        {
            "rallyId": row["rallyId"],
            "recordingId": row["recordingId"],
            "sourceGroup": row["sourceGroup"],
            "environment": row["environment"],
            "decision": row["decision"],
            "probabilityNear": float(probability),
            "prediction": "near" if choice else "far",
            "correct": bool(choice == label),
        }
        for row, probability, choice, label in zip(
            rows, selected_probabilities, predicted, truth, strict=True
        )
    ]
    review = _load(review_path)
    reviewed_rows = review.get("reviewedPredictions")
    if not isinstance(reviewed_rows, list):
        raise ValueError("review slice artifact has no frozen sample")
    reviewed_slices = rescore_reviewed_predictions(reviewed_rows, selected_predictions)
    best_temporal_probabilities = probabilities_by_key[best_temporal_key]
    best_temporal_threshold = float(
        best_temporal["evaluation"]["thresholdSelection"]["threshold"]
    )
    best_temporal_predictions = [
        {
            "rallyId": row["rallyId"],
            "prediction": (
                "near"
                if probability >= best_temporal_threshold
                else "far"
            ),
        }
        for row, probability in zip(
            rows, best_temporal_probabilities, strict=True
        )
    ]
    best_temporal_reviewed_slices = rescore_reviewed_predictions(
        reviewed_rows, best_temporal_predictions
    )
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-temporal-flight-development-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "selection": {
            "primaryMetric": "mean source-group balanced accuracy",
            "tieBreaks": [
                "pooled balanced accuracy",
                "pooled macro-F1",
                "worst source-group balanced accuracy",
                "fewer features",
            ],
            "crossValidation": "leave-one-source-group-out on corrected development rows",
            "protectedTest": "not loaded, scored, or used",
            "reviewSlices": "reported after selection; not a selection metric",
            "selectedCandidate": selected,
            "bestTemporalCandidate": best_temporal,
        },
        "leaderboard": leaderboard,
        "reviewedSliceEstimate": {
            "overall": reviewed_slices["overall"],
            "slices": reviewed_slices["slices"],
        },
        "bestTemporalReviewedSliceEstimate": {
            "overall": best_temporal_reviewed_slices["overall"],
            "slices": best_temporal_reviewed_slices["slices"],
        },
        "finalModel": {
            "fingerprint": fingerprint,
            "featureFamily": selected_key[0],
            "featureNames": selected_names,
            "parameters": final_parameters,
            "trainingRows": len(rows),
        },
        "selectedPredictions": selected_predictions,
        "counts": dataset.get("counts"),
        "sources": {
            "temporalDevelopmentDataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
            },
            "centerFlightDataset": center_source,
            "reviewedSlices": {"path": str(review_path), "sha256": _sha256(review_path)},
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
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"selected {selected_key[0]} / l2={selected_key[1]}")
    print(f"wrote {output_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    result.add_argument("--review-slices", type=Path, default=DEFAULT_REVIEW_SLICES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
