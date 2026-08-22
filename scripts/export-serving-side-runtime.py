#!/usr/bin/env python3
"""Export the frozen serving-side model into the production browser runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SOURCE = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/"
    "serving-side-flight-v3-development.json"
)
DEFAULT_OUTPUT = Path("prod/public/runtime/serving-side-85bc3325fbd4.json")
MODEL_FINGERPRINT = (
    "85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06"
)


def object_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    report = object_value(json.loads(source_bytes), "development report")
    model = object_value(report.get("finalModel"), "finalModel")
    if model.get("fingerprint") != MODEL_FINGERPRINT:
        raise ValueError("source report does not contain the frozen production model")
    feature_names = model.get("featureNames")
    parameters = object_value(model.get("parameters"), "finalModel.parameters")
    if not isinstance(feature_names, list) or len(feature_names) != 237:
        raise ValueError("frozen model must contain the SERVSIDE237-FLIGHT signature")

    artifact = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-fixed-flight-runtime-v1",
        "modelId": "serving-side-fixed-flight-v3",
        "fingerprint": MODEL_FINGERPRINT,
        "sourceReportSha256": hashlib.sha256(source_bytes).hexdigest(),
        "featureVersion": "SERVSIDE237-FLIGHT",
        "courtFlowOffsetsSeconds": [-1.25, -0.75, -0.35, -0.10, 0.10, 0.30, 0.55, 0.85],
        "flightOffsetsSeconds": [-0.15, 0.05, 0.20, 0.35, 0.55, 0.80, 1.10, 1.40, 1.75],
        "resize": {"width": 192, "height": 108},
        "flightGrid": {"rows": 4, "columns": 6},
        "featureNames": feature_names,
        "model": parameters,
        "sideThreshold": 0.4783744762021848,
        "reviewBand": {
            "farUpperExclusive": 0.3121748736511044,
            "nearLowerInclusive": 0.5028396703865513,
        },
        "gate": {
            "serveHeadThreshold": 0.85,
            "serveHeadWindowSeconds": 1.0,
            "rallyRecoveryAgreement": "both-models",
            "rallyRecoveryRequiresReview": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
