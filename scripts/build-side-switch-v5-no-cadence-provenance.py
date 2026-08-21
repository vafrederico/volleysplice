#!/usr/bin/env python3
"""Bind the V5 no-cadence side-switch ablation to one implementation revision."""

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
DEFAULT_INHERITED_PROVENANCE = (
    ROOT / "reports/side-switch/side-switch-production-state-v1-provenance.json"
)
DEFAULT_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v5-production-state-v1-features.json"
)
DEFAULT_CADENCE_MODEL = ROOT / "models/side-switch-v5-production-state-v1/model.json"
DEFAULT_CADENCE_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-production-state-v1-evaluation.json"
)
DEFAULT_MODEL = ROOT / "models/side-switch-v5-no-cadence-v1/model.json"
DEFAULT_DATASET = (
    ROOT / "models/side-switch-v5-no-cadence-v1/dataset-development.json"
)
DEFAULT_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-no-cadence-v1-evaluation.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/side-switch/side-switch-v5-no-cadence-v1-provenance.json"
)
EXPECTED_INHERITED_PROVENANCE_SHA256 = (
    "174fe3980cdd55fa14dda00c7e27d1b01882f303ec3ece6b6d709276e88643e7"
)
IMPLEMENTATION_FILES = (
    Path("analysis/side_switch_no_cadence.py"),
    Path("scripts/train-side-switch-v5-no-cadence.py"),
    Path("scripts/build-side-switch-v5-no-cadence-provenance.py"),
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


def build(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite provenance: {output}")
    revision = subprocess.run(
        ["git", "rev-parse", args.implementation_revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inherited_path = args.inherited_provenance.expanduser().resolve()
    inherited_sha = _sha256(inherited_path)
    if inherited_sha != EXPECTED_INHERITED_PROVENANCE_SHA256:
        raise ValueError("production-state provenance identity changed")
    inherited = _load(inherited_path)
    frozen_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if inherited.get("frozenSplit") != frozen_split:
        raise ValueError("production-state provenance has a different split")

    artifacts = {
        "featureArtifact": args.features.expanduser().resolve(),
        "cadenceModel": args.cadence_model.expanduser().resolve(),
        "cadenceEvaluation": args.cadence_evaluation.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "developmentDataset": args.dataset.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    inherited_artifacts = inherited.get("artifacts")
    if not isinstance(inherited_artifacts, Mapping):
        raise ValueError("production-state provenance has no artifacts")
    inherited_bindings = {
        "featureArtifact": "v5OriginalStateFeatures",
        "cadenceModel": "v5Model",
        "cadenceEvaluation": "v5Evaluation",
    }
    for local_name, inherited_name in inherited_bindings.items():
        inherited_entry = inherited_artifacts.get(inherited_name)
        if (
            not isinstance(inherited_entry, Mapping)
            or inherited_entry.get("sha256") != _sha256(artifacts[local_name])
        ):
            raise ValueError(f"inherited artifact changed: {local_name}")

    model = _load(artifacts["model"])
    evaluation = _load(artifacts["evaluation"])
    if (
        model.get("kind")
        != "volleycut-side-switch-v5-no-cadence-specialist-v1"
        or model.get("dataPolicy", {}).get("frozenRecordingSplit") != frozen_split
        or evaluation.get("kind")
        != "volleycut-side-switch-v5-no-cadence-evaluation-v1"
        or evaluation.get("modelSha256") != _sha256(artifacts["model"])
    ):
        raise ValueError("no-cadence model and evaluation are not mutually bound")

    repository_root = Path.cwd().resolve()
    implementation_files = {
        str(path): {
            "path": str((repository_root / path).resolve()),
            "sha256": _sha256(repository_root / path),
        }
        for path in IMPLEMENTATION_FILES
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-v5-no-cadence-provenance-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementationRevision": revision,
        "frozenSplit": frozen_split,
        "sources": inherited.get("sources"),
        "sourceManifests": {
            "productionStateProvenance": {
                "path": str(inherited_path),
                "sha256": inherited_sha,
                "inheritance": (
                    "reuses the exact 21-video hashes, reviewed-gap identity, split, "
                    "V5 appearance features, and production-state lineage"
                ),
            },
            **dict(inherited.get("sourceManifests", {})),
        },
        "implementationFiles": implementation_files,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "notes": {
            "isolation": (
                "refits the same V5 base and state-gate linear heads, verifies "
                "parameter parity with the cadence variants, and changes only "
                "validation thresholding plus decoder geometry"
            ),
            "decoder": (
                "every frozen reviewed gap is independently selected at or above "
                "the validation threshold; no cadence, spacing, count cap, local "
                "suppression, or re-anchoring is applied"
            ),
            "suppression": "quarantined and never eligible as model input",
            "evaluationStatus": (
                "candidate-conditioned retrospective confirmation, not a pristine test"
            ),
            "promotion": "research diagnostic only; no browser or Android runtime port",
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inherited-provenance", type=Path, default=DEFAULT_INHERITED_PROVENANCE
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--cadence-model", type=Path, default=DEFAULT_CADENCE_MODEL)
    parser.add_argument(
        "--cadence-evaluation", type=Path, default=DEFAULT_CADENCE_EVALUATION
    )
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
                "artifacts": len(payload["artifacts"]),
                "implementationFiles": len(payload["implementationFiles"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
