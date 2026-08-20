#!/usr/bin/env python3
"""Score the frozen serving-side v2 candidate once on protected test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side_specialist import binary_metrics
from analysis.serving_side_v2 import BoostedStumpModel, LogisticModel


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_MODEL = ROOT / "models/serving-side-specialist-v2/model.json"
DEFAULT_DATASET = ROOT / "features/serving-side-v2/protected-test.json"
DEFAULT_OUTPUT = ROOT / "reports/serving-side/serving-side-specialist-v2-protected-test.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def _matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    fields = {"old": "oldFeatures", "flow": "courtFlowFeatures", "rank": "recordingRankFeatures"}
    values = []
    for row in rows:
        vector = []
        for specification in names:
            group, name = specification.split(":", 1)
            value = row.get(fields[group], {}).get(name)
            vector.append(float(value) if isinstance(value, (int, float)) else float("nan"))
        values.append(vector)
    return np.asarray(values, dtype=np.float64)


def _metrics(rows: Sequence[Mapping[str, Any]], predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray([int(row["label"]) for row in rows])
    return binary_metrics(truth, predicted)


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    model_path, dataset_path, output = args.model.resolve(), args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protected evaluation: {output}")
    model_payload, dataset = _load(model_path), _load(dataset_path)
    if model_payload.get("kind") != "volleycut-serving-side-specialist-v2":
        raise ValueError("unexpected model kind")
    if dataset.get("scope") != "protected-test" or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not True:
        raise ValueError("expected a protected-test feature artifact")
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not rows or any(row.get("sourceSplit") != "test" for row in rows):
        raise ValueError("protected dataset contains an invalid scope")
    parameters = model_payload["model"]
    if parameters.get("family") == "class-balanced-logistic":
        model = LogisticModel.from_dict(parameters)
    elif parameters.get("family") == "adaboost-decision-stumps":
        model = BoostedStumpModel.from_dict(parameters)
    else:
        raise ValueError("unknown frozen model family")
    names = model_payload["featureNames"]
    probabilities = model.predict_proba(_matrix(rows, names))
    predicted = probabilities >= model.threshold
    by_recording = {}
    for recording_id in sorted({row["recordingId"] for row in rows}):
        indices = np.asarray([index for index, row in enumerate(rows) if row["recordingId"] == recording_id])
        by_recording[recording_id] = _metrics([rows[index] for index in indices], predicted[indices])
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-specialist-v2-protected-test-evaluation",
        "createdAt": datetime.now(UTC).isoformat(),
        "modelFingerprint": model_payload["fingerprint"],
        "threshold": model.threshold,
        "featureFamily": model_payload["featureFamily"],
        "modelFamily": parameters["family"],
        "metrics": _metrics(rows, predicted),
        "byRecording": by_recording,
        "predictions": [
            {
                "rallyId": row["rallyId"],
                "recordingId": row["recordingId"],
                "environment": row["environment"],
                "split": row["sourceSplit"],
                "decision": row["decision"],
                "nearProbability": float(probability),
                "prediction": "near" if guess else "far",
            }
            for row, probability, guess in zip(rows, probabilities, predicted, strict=True)
        ],
        "sources": {
            "servingSideReport": dataset["sources"]["servingSideReport"],
            "reviewDecisions": dataset["sources"]["reviewDecisions"],
            "frozenModel": {"path": str(model_path), "sha256": _sha256(model_path)},
            "protectedDataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path)},
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "dataPolicy": {
            "protectedTest": "loaded only after the candidate model, feature family, hyperparameters, and threshold were frozen",
            "candidateConditioned": True,
        },
    }
    atomic_write_text(output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}")
    print(json.dumps(result["metrics"], indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
