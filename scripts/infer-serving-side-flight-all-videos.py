#!/usr/bin/env python3
"""Apply the frozen fixed-flight model and review policy to every reviewed video."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side_development_eval import (
    configuration_matrix,
    matrix,
    tied_recording_ranks,
)
from analysis.serving_side_specialist import binary_metrics
from analysis.serving_side_v2 import LogisticModel


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_MODEL_EVALUATION = (
    ROOT / "reports/serving-side/serving-side-flight-v3-development.json"
)
DEFAULT_CALIBRATION = (
    ROOT
    / "reports/serving-side/serving-side-flight-v3-calibration-abstention-development.json"
)
DEFAULT_DEVELOPMENT_FEATURES = (
    ROOT / "features/serving-side-flight-v4/all-reviewed-inference.json"
)
DEFAULT_PROTECTED_FEATURES = ROOT / "features/serving-side-flight-v4/protected-test.json"
DEFAULT_SERVE_EVIDENCE = (
    ROOT
    / "reports/serving-side/serving-side-specialist-v4-dual-serve-gate-all-video-inference-v3.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/serving-side/serving-side-flight-v3-dual-serve-gate-all-video-inference-v1.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_development_eval.py",
    REPOSITORY_ROOT / "analysis/serving_side_v2.py",
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


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _rows(value: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    raw = value.get("rows")
    if not isinstance(raw, list) or not raw or not all(isinstance(row, Mapping) for row in raw):
        raise ValueError(f"{label} rows are unavailable")
    return list(raw)


def _review_decision(probability: float, far: float, near: float) -> str:
    if probability < far:
        return "far"
    if probability >= near:
        return "near"
    return "review"


def _split_metrics(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    result = {}
    for split in sorted({str(row["sourceSplit"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceSplit"] == split]
        )
        truth = np.asarray([int(rows[index]["label"]) for index in indices])
        result[split] = binary_metrics(truth, probabilities[indices] >= threshold)
    return result


def infer(args: argparse.Namespace) -> Mapping[str, Any]:
    model_path = args.model_evaluation.resolve()
    calibration_path = args.calibration.resolve()
    development_path = args.development_features.resolve()
    protected_path = args.protected_features.resolve()
    serve_path = args.serve_evidence.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite all-video inference: {output_path}")
    model_evaluation = _load(model_path)
    calibration = _load(calibration_path)
    development = _load(development_path)
    protected = _load(protected_path)
    serve_evidence = _load(serve_path)
    final_model = model_evaluation.get("finalModel")
    if not isinstance(final_model, Mapping):
        raise ValueError("fixed-flight final model is unavailable")
    fingerprint = str(final_model.get("fingerprint"))
    if (
        calibration.get("modelFingerprint") != fingerprint
        or calibration.get("sources", {}).get("fixedFlightEvaluation", {}).get("sha256")
        != _sha256(model_path)
    ):
        raise ValueError("calibration policy is not bound to the fixed-flight model")
    expected_kinds = {
        development.get("kind"),
        protected.get("kind"),
    }
    if expected_kinds != {"volleycut-serving-side-flight-feature-all-reviewed-inference-v1"}:
        raise ValueError("all-video flight feature banks have unexpected kinds")
    if development.get("dataPolicy", {}).get("protectedTestIncluded") is not False:
        raise ValueError("development inference bank unexpectedly contains test rows")
    if protected.get("dataPolicy", {}).get("protectedTestIncluded") is not True:
        raise ValueError("protected inference bank does not declare its test rows")
    rows = [*_rows(development, "development"), *_rows(protected, "protected")]
    if len({str(row["rallyId"]) for row in rows}) != len(rows):
        raise ValueError("all-video flight banks contain duplicate rally IDs")
    configuration = str(final_model["configuration"])
    v2_names = [str(name) for name in development["v2FeatureNames"]]
    if protected.get("v2FeatureNames") != development.get("v2FeatureNames"):
        raise ValueError("development and protected v2 signatures differ")
    configuration_record = next(
        item
        for item in development["configurations"]
        if isinstance(item, Mapping) and item.get("name") == configuration
    )
    flight_names = [str(name) for name in configuration_record["featureNames"]]
    v2_values = matrix(rows, "v2RecordingRankFeatures", v2_names)
    raw_flight = configuration_matrix(rows, configuration, flight_names)
    flight_ranks = tied_recording_ranks(rows, raw_flight)
    values = np.column_stack((v2_values, flight_ranks))
    expected_names = [*[f"v2:{name}" for name in v2_names], *[f"flight:{name}" for name in flight_names]]
    if expected_names != final_model.get("featureNames"):
        raise ValueError("all-video feature matrix does not match the frozen model signature")
    model = LogisticModel.from_dict(final_model["parameters"])
    probabilities = model.predict_proba(values)
    threshold = float(model.threshold)
    choices = probabilities >= threshold

    old_predictions_value = serve_evidence.get("predictions")
    if not isinstance(old_predictions_value, list) or not all(
        isinstance(row, Mapping) for row in old_predictions_value
    ):
        raise ValueError("serve evidence prediction rows are unavailable")
    old_by_id = {str(row["rallyId"]): row for row in old_predictions_value}
    if set(old_by_id) != {str(row["rallyId"]) for row in rows}:
        raise ValueError("fixed-flight rows do not exactly match the serve-evidence universe")
    recommended = calibration["abstentionSelection"]["recommended"]
    far_threshold = float(recommended["farThreshold"])
    near_threshold = float(recommended["nearThreshold"])
    predictions = []
    for row, probability, choice in zip(rows, probabilities, choices, strict=True):
        rally_id = str(row["rallyId"])
        old = old_by_id[rally_id]
        side = "near" if choice else "far"
        serve_prediction = str(old["servePrediction"])
        if serve_prediction not in {"serve", "not-serve"}:
            raise ValueError(f"invalid serve prediction for {rally_id}")
        predictions.append(
            {
                **old,
                "decision": row["decision"],
                "nearProbability": float(probability),
                "prediction": side,
                "finalPrediction": side if serve_prediction == "serve" else "not-serve",
                "reviewRecommendation": _review_decision(
                    float(probability), far_threshold, near_threshold
                ),
            }
        )
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    metrics = binary_metrics(truth, choices)
    development_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["sourceSplit"] != "test"]
    )
    protected_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["sourceSplit"] == "test"]
    )
    sources = serve_evidence.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("serve evidence source bindings are unavailable")
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-flight-v3-dual-serve-gate-all-video-inference",
        "createdAt": datetime.now(UTC).isoformat(),
        "modelFingerprint": fingerprint,
        "serveGateFingerprint": serve_evidence["serveGateFingerprint"],
        "serveGate": serve_evidence["serveGate"],
        "threshold": threshold,
        "featureFamily": final_model["featureFamily"],
        "modelFamily": final_model["parameters"]["family"],
        "reviewPolicy": {
            "kind": "development-selected-abstention-band",
            "precisionTarget": calibration["abstentionSelection"]["recommendedPrecisionTarget"],
            "farThreshold": far_threshold,
            "nearThreshold": near_threshold,
            "developmentMetrics": recommended,
            "modelFingerprint": fingerprint,
        },
        "metrics": metrics,
        "developmentInSampleMetrics": binary_metrics(
            truth[development_indices], choices[development_indices]
        ),
        "protectedTestMetrics": binary_metrics(
            truth[protected_indices], choices[protected_indices]
        ),
        "bySplit": _split_metrics(rows, probabilities, threshold),
        "counts": {
            "rows": len(rows),
            "recordings": len({str(row["recordingId"]) for row in rows}),
            "developmentRows": len(development_indices),
            "protectedTestRows": len(protected_indices),
            "reviewRecommended": int(
                np.sum(
                    (probabilities >= far_threshold)
                    & (probabilities < near_threshold)
                )
            ),
            "servePredictions": sum(
                prediction["servePrediction"] == "serve" for prediction in predictions
            ),
            "notServePredictions": sum(
                prediction["servePrediction"] == "not-serve" for prediction in predictions
            ),
        },
        "predictions": predictions,
        "dataPolicy": {
            "coverage": "same 1,114 reviewed candidate rows and 30 videos as the prior all-video UI artifact",
            "training": "fixed-flight model fitted only on correction-clean development rows",
            "reviewPolicySelection": "development only; protected rows were not loaded until thresholds were frozen",
            "protectedTest": "post-selection inference only; never used to change the model, calibration, or review band",
            "serveGate": serve_evidence["dataPolicy"]["serveGate"],
        },
        "sources": {
            "servingSideReport": sources["servingSideReport"],
            "reviewDecisions": sources["reviewDecisions"],
            "humanLabelCorrections": sources["humanLabelCorrections"],
            "sourceQualityExclusions": sources["sourceQualityExclusions"],
            "serveGateEvidence": _source(serve_path),
            "fixedFlightEvaluation": _source(model_path),
            "calibrationPolicy": _source(calibration_path),
            "developmentFeatures": _source(development_path),
            "protectedFeatures": _source(protected_path),
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ],
        },
    }
    atomic_write_text(output_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output_path}")
    print(json.dumps({"counts": result["counts"], "metrics": metrics}, indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-evaluation", type=Path, default=DEFAULT_MODEL_EVALUATION)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--development-features", type=Path, default=DEFAULT_DEVELOPMENT_FEATURES)
    parser.add_argument("--protected-features", type=Path, default=DEFAULT_PROTECTED_FEATURES)
    parser.add_argument("--serve-evidence", type=Path, default=DEFAULT_SERVE_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    infer(_parser().parse_args())
