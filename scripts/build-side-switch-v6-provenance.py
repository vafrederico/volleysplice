#!/usr/bin/env python3
"""Publish immutable source, detector, and artifact provenance for side-switch v6."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text
from analysis.side_switch_player_detector import detector_identity
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_V5_PROVENANCE = ROOT / "reports/side-switch/side-switch-v5-provenance.json"
DEFAULT_DETECTOR_DIR = (
    ROOT
    / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
)
DEFAULT_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v6-detected-adaptive-features.json"
)
DEFAULT_MODEL = ROOT / "models/side-switch-specialist-v6-detected-adaptive/model.json"
DEFAULT_DATASET = (
    ROOT
    / "models/side-switch-specialist-v6-detected-adaptive/dataset-development.json"
)
DEFAULT_EVALUATION = (
    ROOT
    / "reports/side-switch/side-switch-specialist-v6-detected-adaptive-evaluation.json"
)
DEFAULT_OUTPUT = ROOT / "reports/side-switch/side-switch-v6-provenance.json"
EXPECTED_V5_PROVENANCE_SHA256 = (
    "479ac49e1aefa0d0e4af841834b9a434575250ec566c58a86e3f57884c01b148"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v6 provenance: {output}")
    revision = subprocess.run(
        ["git", "rev-parse", args.implementation_revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inherited_path = args.v5_provenance.expanduser().resolve()
    inherited_sha = _sha256(inherited_path)
    if inherited_sha != EXPECTED_V5_PROVENANCE_SHA256:
        raise ValueError("v5 provenance identity changed")
    inherited = _load(inherited_path)
    frozen_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if inherited.get("frozenSplit") != frozen_split:
        raise ValueError("v5 provenance does not cover the frozen v6 split")
    inherited_sources = inherited.get("sources")
    if not isinstance(inherited_sources, list) or len(inherited_sources) != 21:
        raise ValueError("v5 provenance does not contain 21 source records")

    detector_dir = args.detector_dir.expanduser().resolve()
    detector = detector_identity(detector_dir)
    artifacts = {
        "featureArtifact": args.features.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "developmentDataset": args.dataset.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
        "personDetectorModel": Path(detector["modelPath"]),
        "personDetectorMetadata": Path(detector["metadataPath"]),
        "personDetectorLicense": Path(detector["licensePath"]),
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-v6-provenance",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementationRevision": revision,
        "frozenSplit": frozen_split,
        "sources": inherited_sources,
        "sourceManifests": {
            "v5Provenance": {
                "path": str(inherited_path),
                "sha256": inherited_sha,
                "inheritance": (
                    "v6 reuses the exact frozen labels, split, v4 court geometry, "
                    "and full-file video hashes bound transitively by v5"
                ),
            },
            **dict(inherited.get("sourceManifests", {})),
        },
        "personDetector": {
            **detector,
            "license": "Apache-2.0",
            "trainingRole": (
                "third-party frozen general-person localizer; no VolleySplice labels "
                "altered detector weights"
            ),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "notes": {
            "videoHash": (
                "full-file SHA-256 inherited transitively from immutable v5/v4/v3 "
                "provenance artifacts"
            ),
            "evaluationStatus": "retrospective, not a pristine promotion test",
            "scope": (
                "v6 replaces motion components with a pinned block-int8 person "
                "localizer and adds confidence-gated online team-color prototypes"
            ),
            "ablation": (
                "model.json contains a separately fitted 26-input detected-player "
                "fixed-prototype ablation alongside the selected 29-input candidate"
            ),
            "promotion": "rejected; v5 remains the best side-switch research artifact",
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-provenance", type=Path, default=DEFAULT_V5_PROVENANCE)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--implementation-revision", default="HEAD")
    return parser


def main() -> None:
    payload = build(_parser().parse_args())
    print(
        json.dumps(
            {
                "implementationRevision": payload["implementationRevision"],
                "sources": len(payload["sources"]),
                "artifacts": len(payload["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
