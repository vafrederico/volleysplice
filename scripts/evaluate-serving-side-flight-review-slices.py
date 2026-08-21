#!/usr/bin/env python3
"""Freeze weighted visibility and contact slices from completed flight reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text
import analysis.serving_side_review_slices as review_slices
from analysis.serving_side_review_slices import build_review_slice_report


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/serving-side"
DEFAULT_EVALUATION = REPORTS / "serving-side-flight-v2-development.json"
DEFAULT_ANNOTATIONS = REPORTS / "serving-side-flight-error-annotations-v2.json"
DEFAULT_COHORT = REPORTS / "serving-side-flight-correct-control-cohort-v1.json"
DEFAULT_CORRECTIONS = REPORTS / "serving-side-result-label-corrections-v1.json"
DEFAULT_OUTPUT = REPORTS / "serving-side-flight-v2-reviewed-slices-v1.json"


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    paths = {
        "evaluation": args.evaluation.resolve(),
        "annotations": args.annotations.resolve(),
        "controlCohort": args.control_cohort.resolve(),
        "humanLabelCorrections": args.corrections.resolve(),
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reviewed slice report: {output}")
    evaluation = _load(paths["evaluation"])
    annotations = _load(paths["annotations"])
    cohort = _load(paths["controlCohort"])
    corrections = _load(paths["humanLabelCorrections"])
    evaluation_sha = _sha256(paths["evaluation"])
    if (
        annotations.get("experimentSha256") != evaluation_sha
        or not isinstance(cohort.get("experiment"), Mapping)
        or cohort["experiment"].get("sha256") != evaluation_sha
    ):
        raise ValueError("review annotations or controls do not match the evaluation")
    report = build_review_slice_report(
        evaluation.get("selectedPredictions", []),
        annotations.get("annotations", {}),
        cohort.get("rows", []),
        corrections.get("corrections", {}),
    )
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-flight-reviewed-slices-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "evaluationKind": evaluation.get("kind"),
        **report,
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "implementation": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (
                Path(__file__).resolve(),
                Path(review_slices.__file__).resolve(),
            )
        ],
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    result.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    result.add_argument("--control-cohort", type=Path, default=DEFAULT_COHORT)
    result.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = evaluate(arguments)
    print(json.dumps({"output": str(arguments.output.resolve()), **result["counts"]}, indent=2))
