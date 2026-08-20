#!/usr/bin/env python3
"""Publish immutable source and artifact provenance for side-switch v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from analysis.artifacts import atomic_write_text
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_V3_PROVENANCE = ROOT / "reports/side-switch/side-switch-v3-provenance.json"
DEFAULT_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v4-multiframe-normalized-features.json"
)
DEFAULT_MODEL = (
    ROOT / "models/side-switch-specialist-v4-multiframe-normalized/model.json"
)
DEFAULT_DATASET = (
    ROOT
    / "models/side-switch-specialist-v4-multiframe-normalized/dataset-development.json"
)
DEFAULT_EVALUATION = (
    ROOT
    / "reports/side-switch/side-switch-specialist-v4-multiframe-normalized-evaluation.json"
)
DEFAULT_OUTPUT = ROOT / "reports/side-switch/side-switch-v4-provenance.json"
EXPECTED_V3_PROVENANCE_SHA256 = (
    "9bcc8d33f3a268f991d311a719205b5657544c23943096629c18e3ca70ba598e"
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
        raise FileExistsError(f"refusing to overwrite v4 provenance: {output}")
    revision = subprocess.run(
        ["git", "rev-parse", args.implementation_revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inherited_path = args.v3_provenance.expanduser().resolve()
    inherited_sha = _sha256(inherited_path)
    if inherited_sha != EXPECTED_V3_PROVENANCE_SHA256:
        raise ValueError("v3 provenance identity changed")
    inherited = _load(inherited_path)
    frozen_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if inherited.get("frozenSplit") != frozen_split:
        raise ValueError("v3 provenance does not cover the frozen v4 split")
    inherited_sources = inherited.get("sources")
    if not isinstance(inherited_sources, list) or len(inherited_sources) != 21:
        raise ValueError("v3 provenance does not contain 21 source records")

    artifacts = {
        "featureArtifact": args.features.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "developmentDataset": args.dataset.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-v4-provenance",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementationRevision": revision,
        "frozenSplit": frozen_split,
        "sources": inherited_sources,
        "sourceManifests": {
            "v3Provenance": {
                "path": str(inherited_path),
                "sha256": inherited_sha,
                "inheritance": (
                    "v4 reuses the exact frozen labels, split, and full-file video "
                    "hashes bound by v3"
                ),
            },
            **dict(inherited.get("sourceManifests", {})),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "notes": {
            "videoHash": (
                "full-file SHA-256 inherited transitively from the immutable v3 "
                "provenance artifact"
            ),
            "evaluationStatus": "retrospective, not a pristine promotion test",
            "scope": (
                "v4 changes only multi-frame side identity, camera-motion "
                "compensation, and court-distance normalization"
            ),
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-provenance", type=Path, default=DEFAULT_V3_PROVENANCE)
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
