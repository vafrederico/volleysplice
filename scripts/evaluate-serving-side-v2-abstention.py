#!/usr/bin/env python3
"""Select a two-threshold serving-side abstention policy on development data."""

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
from analysis.serving_side_v2 import fit_boosted_stumps, fit_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_MODEL = ROOT / "models/serving-side-specialist-v2/model.json"
DEFAULT_DATASET = ROOT / "features/serving-side-v2/development.json"
DEFAULT_OUTPUT = (
    ROOT
    / "reports/serving-side/serving-side-specialist-v2-development-abstention.json"
)
PRECISION_TARGETS = (0.90, 0.925, 0.95, 0.975)
RECOMMENDED_PRECISION_TARGET = 0.95
DEFAULT_MINIMUM_PREDICTIONS_PER_SIDE = 50
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_abstention.py",
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


def _matrix(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> np.ndarray:
    fields = {
        "old": "oldFeatures",
        "flow": "courtFlowFeatures",
        "rank": "recordingRankFeatures",
    }
    values = []
    for row in rows:
        vector = []
        for specification in names:
            group, name = specification.split(":", 1)
            raw = row.get(fields[group], {}).get(name)
            vector.append(
                float(raw) if isinstance(raw, (int, float)) else float("nan")
            )
        values.append(vector)
    return np.asarray(values, dtype=np.float64)


def _fit(
    specification: Mapping[str, Any], values: np.ndarray, truth: np.ndarray
) -> Any:
    if specification.get("modelFamily") == "logistic":
        return fit_logistic(values, truth, l2=float(specification["l2"]))
    if specification.get("modelFamily") == "boosted-stumps":
        return fit_boosted_stumps(
            values,
            truth,
            estimators=int(specification["estimators"]),
            learning_rate=float(specification["learningRate"]),
        )
    raise ValueError("unknown selected development model family")


def _cross_fit(
    rows: list[Mapping[str, Any]],
    names: Sequence[str],
    specification: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    values = _matrix(rows, names)
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.full(len(rows), np.nan)
    folds = []
    for group in groups:
        held = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == group]
        )
        train = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] != group]
        )
        model = _fit(specification, values[train], truth[train])
        probabilities[held] = model.predict_proba(values[held])
        folds.append(
            {
                "heldSourceGroup": group,
                "trainRows": len(train),
                "heldRows": len(held),
            }
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("cross-fit left development rows unscored")
    return probabilities, folds


def _decision(probability: float, operating_point: Mapping[str, Any]) -> str:
    if probability < float(operating_point["farThreshold"]):
        return "far"
    if probability >= float(operating_point["nearThreshold"]):
        return "near"
    return "abstain"


def _group_metrics(
    rows: list[Mapping[str, Any]],
    probabilities: np.ndarray,
    operating_point: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for group in sorted({str(row["sourceGroup"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == group]
        )
        result[group] = selective_metrics(
            [int(rows[index]["label"]) for index in indices],
            probabilities[indices],
            far_threshold=float(operating_point["farThreshold"]),
            near_threshold=float(operating_point["nearThreshold"]),
        )
    return result


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    model_path = args.model.resolve()
    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite abstention evaluation: {output_path}")
    model_payload = _load(model_path)
    dataset = _load(dataset_path)
    if model_payload.get("kind") != "volleycut-serving-side-specialist-v2":
        raise ValueError("unexpected model kind")
    expected_dataset = model_payload.get("sources", {}).get("developmentDataset", {})
    if expected_dataset.get("sha256") != _sha256(dataset_path):
        raise ValueError("development dataset does not match the frozen model")
    if (
        dataset.get("scope") != "development"
        or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False
    ):
        raise ValueError("abstention selection requires development-only data")
    raw_rows = dataset.get("rows")
    if (
        not isinstance(raw_rows, list)
        or not raw_rows
        or not all(isinstance(row, Mapping) for row in raw_rows)
        or any(row.get("sourceSplit") == "test" for row in raw_rows)
    ):
        raise ValueError("development rows are empty, malformed, or contain test data")
    rows = list(raw_rows)
    names = model_payload.get("featureNames")
    selected = model_payload.get("selection", {}).get("selectedCandidate")
    if not isinstance(names, list) or not names or not isinstance(selected, Mapping):
        raise ValueError("frozen model is missing its selected feature/model contract")

    probabilities, folds = _cross_fit(rows, names, selected)
    prediction_digest = hashlib.sha256(probabilities.tobytes()).hexdigest()
    if prediction_digest != selected.get("predictionDigest"):
        raise ValueError("development cross-fit predictions do not reproduce the frozen selection")
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    center_threshold = float(model_payload.get("model", {}).get("threshold"))
    baseline = selective_metrics(
        truth,
        probabilities,
        far_threshold=center_threshold,
        near_threshold=center_threshold,
    )
    minimum_predictions = int(args.minimum_predictions_per_side)
    if minimum_predictions < 1:
        raise ValueError("minimum predictions per side must be positive")
    target_points = {
        f"{target:.3f}": select_precision_operating_point(
            truth,
            probabilities,
            center_threshold=center_threshold,
            precision_floor=target,
            minimum_predictions_per_side=minimum_predictions,
        )
        for target in PRECISION_TARGETS
    }
    recommended = target_points[f"{RECOMMENDED_PRECISION_TARGET:.3f}"]
    symmetric_curve = []
    for step in range(19):
        margin = step * 0.025
        symmetric_curve.append(
            {
                "margin": margin,
                **selective_metrics(
                    truth,
                    probabilities,
                    far_threshold=max(0.0, center_threshold - margin),
                    near_threshold=min(1.0, center_threshold + margin),
                ),
            }
        )
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-specialist-v2-development-abstention",
        "createdAt": datetime.now(UTC).isoformat(),
        "modelFingerprint": model_payload["fingerprint"],
        "scope": "development-cross-fit",
        "method": "leave-one-source-group-out",
        "featureFamily": model_payload["featureFamily"],
        "modelFamily": model_payload["model"]["family"],
        "centerThreshold": center_threshold,
        "minimumPredictionsPerSide": minimum_predictions,
        "baseline": baseline,
        "precisionTargetOperatingPoints": target_points,
        "recommendedPrecisionTarget": RECOMMENDED_PRECISION_TARGET,
        "recommended": recommended,
        "recommendedBySourceGroup": (
            _group_metrics(rows, probabilities, recommended) if recommended else None
        ),
        "symmetricMarginCurve": symmetric_curve,
        "folds": folds,
        "predictions": [
            {
                "rallyId": row["rallyId"],
                "recordingId": row["recordingId"],
                "sourceGroup": row["sourceGroup"],
                "environment": row["environment"],
                "decision": row["decision"],
                "nearProbability": float(probability),
                "selectivePrediction": (
                    _decision(float(probability), recommended)
                    if recommended
                    else None
                ),
            }
            for row, probability in zip(rows, probabilities, strict=True)
        ],
        "selectionPolicy": {
            "constraint": "both pooled class precisions meet the target",
            "primary": "maximum emitted-row coverage",
            "tieBreaks": [
                "higher worst-class all-row recall",
                "higher balanced all-row recall",
                "higher selective accuracy",
                "narrower abstention band",
            ],
            "thresholdConstraint": "far threshold <= frozen threshold <= near threshold; existing side decisions can only become abstentions",
        },
        "sources": {
            "frozenModel": {"path": str(model_path), "sha256": _sha256(model_path)},
            "developmentDataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
            },
            "developmentPredictionDigest": prediction_digest,
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ],
        },
        "dataPolicy": {
            "protectedTestLoaded": False,
            "protectedTestUsedForSelection": False,
            "candidateConditioned": True,
            "recallDenominator": "all human rows of that side, including abstained rows",
            "precisionDenominator": "emitted predictions of that side only",
        },
    }
    atomic_write_text(
        output_path, json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {output_path}")
    print(json.dumps({"baseline": baseline, "recommended": recommended}, indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--minimum-predictions-per-side",
        type=int,
        default=DEFAULT_MINIMUM_PREDICTIONS_PER_SIDE,
    )
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
