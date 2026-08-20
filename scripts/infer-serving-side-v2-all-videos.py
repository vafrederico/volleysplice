#!/usr/bin/env python3
"""Run the frozen serving-side v2 model across every available reviewed video."""

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
DEFAULT_DEVELOPMENT = ROOT / "features/serving-side-v2/development.json"
DEFAULT_PROTECTED = ROOT / "features/serving-side-v2/protected-test.json"
DEFAULT_OUTPUT = (
    ROOT
    / "reports/serving-side/serving-side-specialist-v2-all-video-inference.json"
)
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


def _matrix(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> np.ndarray:
    fields = {
        "old": "oldFeatures",
        "flow": "courtFlowFeatures",
        "rank": "recordingRankFeatures",
    }
    values: list[list[float]] = []
    for row in rows:
        vector: list[float] = []
        for specification in names:
            group, name = specification.split(":", 1)
            raw = row.get(fields[group], {}).get(name)
            vector.append(
                float(raw) if isinstance(raw, (int, float)) else float("nan")
            )
        values.append(vector)
    return np.asarray(values, dtype=np.float64)


def _metrics(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray([int(row["label"]) for row in rows])
    return binary_metrics(truth, predicted)


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, field: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in sorted({str(row[field]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row[field] == group]
        )
        result[group] = _metrics(
            [rows[index] for index in indices], predicted[indices]
        )
    return result


def infer(args: argparse.Namespace) -> Mapping[str, Any]:
    model_path = args.model.resolve()
    development_path = args.development_dataset.resolve()
    protected_path = args.protected_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite all-video inference: {output}")

    model_payload = _load(model_path)
    development = _load(development_path)
    protected = _load(protected_path)
    if model_payload.get("kind") != "volleycut-serving-side-specialist-v2":
        raise ValueError("unexpected frozen model kind")
    if development.get("scope") != "development":
        raise ValueError("expected the development feature bank")
    if protected.get("scope") != "protected-test":
        raise ValueError("expected the protected-test feature bank")
    if development.get("featureVersion") != protected.get("featureVersion"):
        raise ValueError("feature banks use different extractor versions")
    if development.get("featureNames") != protected.get("featureNames"):
        raise ValueError("feature banks use different ordered feature signatures")
    for source in ("servingSideReport", "reviewDecisions"):
        if development["sources"][source]["sha256"] != protected["sources"][source]["sha256"]:
            raise ValueError(f"feature banks bind different {source} artifacts")

    development_rows = development.get("rows")
    protected_rows = protected.get("rows")
    if not isinstance(development_rows, list) or not isinstance(protected_rows, list):
        raise ValueError("feature banks contain invalid rows")
    if any(row.get("sourceSplit") == "test" for row in development_rows):
        raise ValueError("protected-test row found in development feature bank")
    if any(row.get("sourceSplit") != "test" for row in protected_rows):
        raise ValueError("non-test row found in protected feature bank")
    rows = sorted(
        [*development_rows, *protected_rows],
        key=lambda row: (
            str(row["recordingId"]),
            float(row["serveAnchor"]),
            str(row["rallyId"]),
        ),
    )
    rally_ids = [str(row["rallyId"]) for row in rows]
    if len(set(rally_ids)) != len(rally_ids):
        raise ValueError("duplicate rally IDs found across feature banks")

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
    development_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["sourceSplit"] != "test"]
    )
    protected_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["sourceSplit"] == "test"]
    )
    recording_ids = sorted({str(row["recordingId"]) for row in rows})
    if len(recording_ids) != 30:
        raise ValueError(f"expected all 30 videos, found {len(recording_ids)}")

    sources = {
        "servingSideReport": development["sources"]["servingSideReport"],
        "reviewDecisions": development["sources"]["reviewDecisions"],
        "frozenModel": {"path": str(model_path), "sha256": _sha256(model_path)},
        "developmentDataset": {
            "path": str(development_path),
            "sha256": _sha256(development_path),
        },
        "protectedDataset": {
            "path": str(protected_path),
            "sha256": _sha256(protected_path),
        },
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-specialist-v2-all-video-inference",
        "createdAt": datetime.now(UTC).isoformat(),
        "modelFingerprint": model_payload["fingerprint"],
        "threshold": model.threshold,
        "featureFamily": model_payload["featureFamily"],
        "modelFamily": parameters["family"],
        "metrics": _metrics(rows, predicted),
        "developmentInSampleMetrics": _metrics(
            [rows[index] for index in development_indices],
            predicted[development_indices],
        ),
        "protectedTestMetrics": _metrics(
            [rows[index] for index in protected_indices],
            predicted[protected_indices],
        ),
        "bySplit": _group_metrics(rows, predicted, "sourceSplit"),
        "byRecording": _group_metrics(rows, predicted, "recordingId"),
        "counts": {
            "rows": len(rows),
            "recordings": len(recording_ids),
            "recordingIds": recording_ids,
            "developmentRows": len(development_indices),
            "protectedTestRows": len(protected_indices),
        },
        "predictions": [
            {
                "rallyId": row["rallyId"],
                "recordingId": row["recordingId"],
                "environment": row["environment"],
                "split": row["sourceSplit"],
                "decision": row["decision"],
                "nearProbability": float(probability),
                "prediction": "near" if guess else "far",
                "evaluationRole": (
                    "protected-test"
                    if row["sourceSplit"] == "test"
                    else "development-in-sample"
                ),
            }
            for row, probability, guess in zip(
                rows, probabilities, predicted, strict=True
            )
        ],
        "dataPolicy": {
            "coverage": "every video with at least one clear near/far review decision",
            "unclear": "excluded because the frozen binary model has no unclear class",
            "development": "predictions are from the final model fitted on development; metrics are in-sample diagnostics",
            "protectedTest": "predictions retain the single frozen held-out evaluation",
            "candidateConditioned": True,
        },
        "sources": sources,
    }
    atomic_write_text(output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}")
    print(
        f"videos={len(recording_ids)} rows={len(rows)} "
        f"development={len(development_indices)} protected={len(protected_indices)}"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--development-dataset", type=Path, default=DEFAULT_DEVELOPMENT
    )
    parser.add_argument("--protected-dataset", type=Path, default=DEFAULT_PROTECTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    infer(_parser().parse_args())
