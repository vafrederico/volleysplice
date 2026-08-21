#!/usr/bin/env python3
"""Record a bulk human confirmation of the frozen serving-side uncertainty queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_INFERENCE = (
    ROOT
    / "reports/serving-side/serving-side-flight-v3-dual-serve-gate-all-video-inference-v1.json"
)
DEFAULT_DEVELOPMENT = ROOT / "features/serving-side-flight-v3/development.json"
DEFAULT_OUTPUT = (
    ROOT
    / "reports/serving-side/serving-side-flight-v3-uncertainty-review-v1.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def _bound_source(
    inference: Mapping[str, Any], name: str
) -> tuple[Path, Mapping[str, Any]]:
    source = inference.get("sources", {}).get(name)
    if not isinstance(source, Mapping):
        raise ValueError(f"inference lacks its {name} binding")
    path = Path(str(source.get("path"))).resolve()
    if _sha256(path) != source.get("sha256"):
        raise ValueError(f"inference {name} source has changed")
    return path, _load(path)


def record(args: argparse.Namespace) -> Mapping[str, Any]:
    inference_path = args.inference.resolve()
    development_path = args.development.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite uncertainty review: {output_path}")
    inference = _load(inference_path)
    if (
        inference.get("kind")
        != "volleycut-serving-side-flight-v3-dual-serve-gate-all-video-inference"
        or not isinstance(inference.get("reviewPolicy"), Mapping)
    ):
        raise ValueError("input is not the frozen fixed-flight all-video inference")
    decisions_path, decisions = _bound_source(inference, "reviewDecisions")
    corrections_path, corrections = _bound_source(
        inference, "humanLabelCorrections"
    )
    report_path, report = _bound_source(inference, "servingSideReport")
    raw_decisions = decisions.get("decisions")
    raw_corrections = corrections.get("corrections")
    if not isinstance(raw_decisions, Mapping) or not isinstance(
        raw_corrections, Mapping
    ):
        raise ValueError("current human labels are unavailable")
    raw_rallies = report.get("rallies")
    if not isinstance(raw_rallies, list) or not all(
        isinstance(row, Mapping) for row in raw_rallies
    ):
        raise ValueError("serving-side rally metadata is unavailable")
    rally_by_id = {str(row["rallyId"]): row for row in raw_rallies}
    development = _load(development_path)
    if (
        development.get("scope") != "development"
        or development.get("dataPolicy", {}).get("protectedTestIncluded") is not False
    ):
        raise ValueError("training eligibility requires development-only features")
    development_rows = development.get("rows")
    if not isinstance(development_rows, list):
        raise ValueError("development feature rows are unavailable")
    development_ids = {
        str(row["rallyId"])
        for row in development_rows
        if isinstance(row, Mapping) and isinstance(row.get("rallyId"), str)
    }
    raw_predictions = inference.get("predictions")
    if not isinstance(raw_predictions, list) or not all(
        isinstance(row, Mapping) for row in raw_predictions
    ):
        raise ValueError("inference predictions are unavailable")
    uncertain = [
        row for row in raw_predictions if row.get("reviewRecommendation") == "review"
    ]
    expected_count = inference.get("counts", {}).get("reviewRecommended")
    if not uncertain or len(uncertain) != expected_count:
        raise ValueError("inference uncertainty count is inconsistent")
    reviewed_at = datetime.now(UTC).isoformat()
    rows = []
    for prediction in uncertain:
        rally_id = str(prediction["rallyId"])
        original = raw_decisions.get(rally_id)
        human = raw_corrections.get(rally_id, original)
        if original not in {"near", "far"} or human not in {
            "near",
            "far",
            "not-serve",
        }:
            raise ValueError(f"current human label is unavailable for {rally_id}")
        rally = rally_by_id.get(rally_id)
        if not isinstance(rally, Mapping):
            raise ValueError(f"rally metadata is unavailable for {rally_id}")
        rows.append(
            {
                "rallyId": rally_id,
                "recordingId": prediction["recordingId"],
                "sourceGroup": rally["sourceGroup"],
                "split": prediction["split"],
                "currentHumanLabel": human,
                "humanLabelCorrect": True,
                "modelSidePrediction": prediction["prediction"],
                "modelFinalPrediction": prediction["finalPrediction"],
                "servePrediction": prediction["servePrediction"],
                "nearProbability": prediction["nearProbability"],
                "trainingEligible": (
                    rally_id in development_ids and human in {"near", "far"}
                ),
                "reviewedAt": reviewed_at,
            }
        )
    rows.sort(key=lambda row: str(row["rallyId"]))
    training_eligible = sum(bool(row["trainingEligible"]) for row in rows)
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-flight-v3-uncertainty-review-v1",
        "createdAt": reviewed_at,
        "reviewStatus": "complete",
        "reviewMethod": "bulk human confirmation after inspecting the model-results UI",
        "modelFingerprint": inference["modelFingerprint"],
        "reviewPolicy": inference["reviewPolicy"],
        "counts": {
            "confirmed": len(rows),
            "near": sum(row["currentHumanLabel"] == "near" for row in rows),
            "far": sum(row["currentHumanLabel"] == "far" for row in rows),
            "notServe": sum(
                row["currentHumanLabel"] == "not-serve" for row in rows
            ),
            "trainingEligible": training_eligible,
            "trainingIneligible": len(rows) - training_eligible,
            "protectedTest": sum(row["split"] == "test" for row in rows),
        },
        "rows": rows,
        "sources": {
            "allVideoInference": {
                "path": str(inference_path),
                "sha256": _sha256(inference_path),
            },
            "reviewDecisions": {
                "path": str(decisions_path),
                "sha256": _sha256(decisions_path),
            },
            "servingSideReport": {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            },
            "humanLabelCorrections": {
                "path": str(corrections_path),
                "sha256": _sha256(corrections_path),
            },
            "developmentFeatures": {
                "path": str(development_path),
                "sha256": _sha256(development_path),
            },
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "dataPolicy": {
            "labelMutation": False,
            "reviewWasBlind": False,
            "modelResultsVisibleToReviewer": True,
            "developmentRows": "confirmation provenance may be consumed by future feature/model pipelines",
            "protectedRows": "verification only; prohibited from training, calibration, thresholding, or model selection",
        },
    }
    atomic_write_text(output_path, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output_path}")
    print(json.dumps(result["counts"], indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    record(_parser().parse_args())
