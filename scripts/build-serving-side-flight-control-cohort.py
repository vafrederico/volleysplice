#!/usr/bin/env python3
"""Freeze a representative correct-prediction cohort for visibility review."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text
from analysis import serving_side_control_cohort
from analysis.serving_side_control_cohort import build_correct_control_cohort


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_EVALUATION = (
    ROOT / "reports/serving-side/serving-side-flight-v2-development.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/serving-side/serving-side-flight-correct-control-cohort-v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build(args: argparse.Namespace) -> Mapping[str, Any]:
    evaluation_path = args.evaluation.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen cohort: {output_path}")
    evaluation = _load(evaluation_path)
    if (
        evaluation.get("schemaVersion") != 1
        or evaluation.get("kind")
        != "volleycut-serving-side-flight-development-evaluation-v1"
    ):
        raise ValueError("input is not a serving-side flight development evaluation")
    predictions = evaluation.get("selectedPredictions")
    selection = evaluation.get("selection")
    selected = selection.get("selectedCandidate") if isinstance(selection, Mapping) else None
    selected_evaluation = selected.get("evaluation") if isinstance(selected, Mapping) else None
    prediction_digest = (
        selected_evaluation.get("predictionDigest")
        if isinstance(selected_evaluation, Mapping)
        else None
    )
    if not isinstance(predictions, list) or not isinstance(prediction_digest, str):
        raise ValueError("evaluation lacks selected predictions or their digest")
    evaluation_sha256 = _sha256(evaluation_path)
    cohort = build_correct_control_cohort(
        predictions,
        experiment_sha256=evaluation_sha256,
        target_rows=args.target_rows,
    )
    payload = {
        "schemaVersion": 1,
        **cohort,
        "createdAt": datetime.now(UTC).isoformat(),
        "experiment": {
            "path": str(evaluation_path),
            "sha256": evaluation_sha256,
            "kind": evaluation["kind"],
            "createdAt": evaluation.get("createdAt"),
            "predictionDigest": prediction_digest,
        },
        "rows": cohort["rows"],
        "implementation": {
            "files": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in (
                    Path(__file__).resolve(),
                    Path(serving_side_control_cohort.__file__).resolve(),
                )
            ]
        },
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--target-rows", type=int, default=120)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    artifact = build(arguments)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "populationRows": artifact["sampling"]["populationRows"],
                "sampledRows": artifact["sampling"]["sampledRows"],
                "strata": len(artifact["sampling"]["strata"]),
            },
            indent=2,
        )
    )
