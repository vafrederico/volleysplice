#!/usr/bin/env python3
"""Select and freeze serving-side v2 on development data only."""

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
from analysis.serving_side_specialist import FULL_EXISTING_FEATURES, binary_metrics, select_threshold
from analysis.serving_side_v2 import FEATURE_NAMES, fit_boosted_stumps, fit_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_DATASET = ROOT / "features/serving-side-v2/development.json"
DEFAULT_MODEL_DIR = ROOT / "models/serving-side-specialist-v2"
DEFAULT_EVALUATION = ROOT / "reports/serving-side/serving-side-specialist-v2-development.json"
L2_GRID = (0.01, 0.1, 1.0, 10.0)
BOOST_GRID = ((25, 0.1), (50, 0.1))
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_v2.py",
    REPOSITORY_ROOT / "scripts/extract-serving-side-v2-features.py",
    Path(__file__).resolve(),
)
FEATURE_FAMILIES = {
    "existing-v1": tuple(f"old:{name}" for name in FULL_EXISTING_FEATURES),
    "court-flow-absolute": tuple(f"flow:{name}" for name in FEATURE_NAMES),
    "court-flow-recording-rank": tuple(f"rank:{name}" for name in FEATURE_NAMES),
    "existing-plus-court-flow": (
        *(f"old:{name}" for name in FULL_EXISTING_FEATURES),
        *(f"flow:{name}" for name in FEATURE_NAMES),
    ),
    "existing-plus-court-flow-rank": (
        *(f"old:{name}" for name in FULL_EXISTING_FEATURES),
        *(f"rank:{name}" for name in FEATURE_NAMES),
    ),
}


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


def _value(row: Mapping[str, Any], specification: str) -> float:
    group, name = specification.split(":", 1)
    field = {"old": "oldFeatures", "flow": "courtFlowFeatures", "rank": "recordingRankFeatures"}[group]
    raw = row.get(field, {}).get(name)
    return float(raw) if isinstance(raw, (int, float)) else float("nan")


def matrix(rows: Sequence[Mapping[str, Any]], family: str) -> np.ndarray:
    names = FEATURE_FAMILIES[family]
    return np.asarray([[_value(row, name) for name in names] for row in rows], dtype=np.float64)


def labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def _fit(specification: Mapping[str, Any], values: np.ndarray, truth: np.ndarray) -> Any:
    if specification["modelFamily"] == "logistic":
        return fit_logistic(values, truth, l2=float(specification["l2"]))
    return fit_boosted_stumps(
        values,
        truth,
        estimators=int(specification["estimators"]),
        learning_rate=float(specification["learningRate"]),
    )


def _group_metrics(rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, key: str) -> dict[str, Any]:
    result = {}
    for group in sorted({str(row[key]) for row in rows}):
        indices = np.asarray([index for index, row in enumerate(rows) if row[key] == group])
        result[group] = binary_metrics(labels([rows[index] for index in indices]), predicted[indices])
    return result


def _cross_fit(rows: list[Mapping[str, Any]], specification: Mapping[str, Any]) -> dict[str, Any]:
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("source-group cross-fit needs at least three groups")
    values = matrix(rows, str(specification["featureFamily"]))
    truth = labels(rows)
    probabilities = np.full(len(rows), np.nan)
    folds = []
    for group in groups:
        held = np.asarray([index for index, row in enumerate(rows) if row["sourceGroup"] == group])
        train = np.asarray([index for index, row in enumerate(rows) if row["sourceGroup"] != group])
        model = _fit(specification, values[train], truth[train])
        probabilities[held] = model.predict_proba(values[held])
        folds.append({"heldSourceGroup": group, "trainRows": len(train), "heldRows": len(held)})
    if not np.isfinite(probabilities).all():
        raise AssertionError("cross-fit left rows unscored")
    threshold_metrics = select_threshold(truth, probabilities)
    threshold = float(threshold_metrics["threshold"])
    predicted = probabilities >= threshold
    by_group = _group_metrics(rows, predicted, "sourceGroup")
    balanced = [float(value["balancedAccuracy"]) for value in by_group.values() if value["balancedAccuracy"] is not None]
    return {
        "thresholdSelection": threshold_metrics,
        "pooledMetrics": binary_metrics(truth, predicted),
        "sourceGroupMacroBalancedAccuracy": float(np.mean(balanced)),
        "worstSourceGroupBalancedAccuracy": float(np.min(balanced)),
        "bySourceGroup": by_group,
        "folds": folds,
        "probabilities": probabilities,
    }


def _candidate_specs() -> list[dict[str, Any]]:
    candidates = [
        {"featureFamily": family, "modelFamily": "logistic", "l2": l2}
        for family in FEATURE_FAMILIES
        for l2 in L2_GRID
    ]
    for family in ("existing-v1", "court-flow-recording-rank", "existing-plus-court-flow-rank"):
        candidates.extend(
            {
                "featureFamily": family,
                "modelFamily": "boosted-stumps",
                "estimators": estimators,
                "learningRate": learning_rate,
            }
            for estimators, learning_rate in BOOST_GRID
        )
    return candidates


def _rank(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    evaluation = candidate["developmentCrossFit"]
    metrics = evaluation["pooledMetrics"]
    return (
        float(evaluation["sourceGroupMacroBalancedAccuracy"]),
        float(metrics["balancedAccuracy"] or 0),
        float(metrics["macroF1"] or 0),
        float(evaluation["worstSourceGroupBalancedAccuracy"]),
        -len(FEATURE_FAMILIES[str(candidate["featureFamily"])]),
    )


def train(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset_path = args.dataset.resolve()
    model_dir = args.model_dir.resolve()
    evaluation_path = args.evaluation_output.resolve()
    if model_dir.exists() or evaluation_path.exists():
        raise FileExistsError("refusing to overwrite a frozen v2 model or evaluation")
    dataset = _load(dataset_path)
    if dataset.get("scope") != "development" or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False:
        raise ValueError("model selection requires a development-only feature artifact")
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not rows or any(row.get("sourceSplit") == "test" for row in rows):
        raise ValueError("development rows are empty or contain protected-test data")
    candidates = []
    for index, specification in enumerate(_candidate_specs(), start=1):
        print(f"candidate {index}/{len(_candidate_specs())}: {specification}", flush=True)
        audit = _cross_fit(rows, specification)
        probabilities = audit.pop("probabilities")
        candidates.append({
            **specification,
            "developmentCrossFit": audit,
            "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
        })
    candidates.sort(key=_rank, reverse=True)
    selected = candidates[0]
    selected_family = str(selected["featureFamily"])
    model = _fit(selected, matrix(rows, selected_family), labels(rows))
    model = replace(model, threshold=float(selected["developmentCrossFit"]["thresholdSelection"]["threshold"]))
    parameters = model.to_dict()
    fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    created = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-specialist-v2",
        "createdAt": created,
        "fingerprint": fingerprint,
        "featureFamily": selected_family,
        "featureNames": list(FEATURE_FAMILIES[selected_family]),
        "model": parameters,
        "selection": {
            "primaryMetric": "mean source-group balanced accuracy",
            "tieBreaks": ["pooled balanced accuracy", "pooled macro-F1", "worst source-group balanced accuracy", "fewer features"],
            "crossValidation": "leave-one-source-group-out over all non-test clear reviews",
            "selectedCandidate": selected,
        },
        "dataPolicy": {
            "development": "all clear reviewed decisions except split=test",
            "protectedTest": "not loaded, scored, or used by this script",
            "previouslyOpenedScopes": "challenge and non-training results were opened in v1 and are development data for v2; they are not unbiased v2 evaluation",
            "unclear": "excluded",
        },
        "sources": {
            "developmentDataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path)},
            "featurePipeline": {
                "version": dataset.get("featureVersion"),
                "featureNames": dataset.get("featureNames"),
                "offsetsSeconds": dataset.get("offsetsSeconds"),
                "resize": dataset.get("resize"),
            },
            "implementation": [
                {"path": str(path.relative_to(REPOSITORY_ROOT)), "sha256": _sha256(path)}
                for path in IMPLEMENTATION_PATHS
            ],
        },
    }
    leaderboard = []
    for candidate in candidates:
        leaderboard.append(candidate)
    evaluation = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-specialist-v2-development-evaluation",
        "createdAt": created,
        "modelFingerprint": fingerprint,
        "modelPath": str(model_dir / "model.json"),
        "selected": selected,
        "leaderboard": leaderboard,
        "counts": dataset.get("counts"),
        "sources": model_payload["sources"],
    }
    atomic_write_text(model_dir / "model.json", json.dumps(model_payload, indent=2, allow_nan=False) + "\n")
    atomic_write_text(evaluation_path, json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
    print(f"selected {selected_family} / {parameters['family']} ({fingerprint[:12]})")
    print(f"wrote {model_dir / 'model.json'}")
    return evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVALUATION)
    return parser


if __name__ == "__main__":
    train(_parser().parse_args())
