#!/usr/bin/env python3
"""Bind the V5 peak/count/context cleanup experiment to one revision."""

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
    ROOT / "reports/side-switch/side-switch-v5-no-cadence-v1-provenance.json"
)
DEFAULT_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v5-production-state-v1-features.json"
)
DEFAULT_NO_CADENCE_MODEL = ROOT / "models/side-switch-v5-no-cadence-v1/model.json"
DEFAULT_NO_CADENCE_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-no-cadence-v1-evaluation.json"
)
DEFAULT_CADENCE_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-production-state-v1-evaluation.json"
)
DEFAULT_MODEL = ROOT / "models/side-switch-v5-peak-cleanup-v1/model.json"
DEFAULT_DATASET = (
    ROOT / "models/side-switch-v5-peak-cleanup-v1/dataset-development.json"
)
DEFAULT_EVALUATION = (
    ROOT / "reports/side-switch/side-switch-v5-peak-cleanup-v1-evaluation.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/side-switch/side-switch-v5-peak-cleanup-v1-provenance.json"
)
EXPECTED_INHERITED_PROVENANCE_SHA256 = (
    "b0a6d21df15c4c5f7f968efa600836bd1b13c6d72791ce746c0c89c7725d64e2"
)
IMPLEMENTATION_FILES = (
    Path("analysis/side_switch_peak_cleanup.py"),
    Path("scripts/train-side-switch-v5-peak-cleanup.py"),
    Path("scripts/build-side-switch-v5-peak-cleanup-provenance.py"),
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
        raise ValueError("no-cadence provenance identity changed")
    inherited = _load(inherited_path)
    frozen_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if inherited.get("frozenSplit") != frozen_split:
        raise ValueError("no-cadence provenance has a different split")

    artifacts = {
        "featureArtifact": args.features.expanduser().resolve(),
        "noCadenceModel": args.no_cadence_model.expanduser().resolve(),
        "noCadenceEvaluation": args.no_cadence_evaluation.expanduser().resolve(),
        "cadenceEvaluation": args.cadence_evaluation.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "developmentDataset": args.dataset.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    inherited_artifacts = inherited.get("artifacts")
    if not isinstance(inherited_artifacts, Mapping):
        raise ValueError("no-cadence provenance has no artifacts")
    inherited_bindings = {
        "featureArtifact": "featureArtifact",
        "noCadenceModel": "model",
        "noCadenceEvaluation": "evaluation",
        "cadenceEvaluation": "cadenceEvaluation",
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
        != "volleycut-side-switch-v5-peak-cleanup-specialist-v1"
        or model.get("dataPolicy", {}).get("frozenRecordingSplit") != frozen_split
        or evaluation.get("kind")
        != "volleycut-side-switch-v5-peak-cleanup-evaluation-v1"
        or evaluation.get("modelSha256") != _sha256(artifacts["model"])
    ):
        raise ValueError("peak-cleanup model and evaluation are not mutually bound")

    repository_root = Path.cwd().resolve()
    implementation_files = {
        str(relative_path): {
            "path": str((repository_root / relative_path).resolve()),
            "sha256": _sha256(repository_root / relative_path),
        }
        for relative_path in IMPLEMENTATION_FILES
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-v5-peak-cleanup-provenance-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementationRevision": revision,
        "frozenSplit": frozen_split,
        "sources": inherited.get("sources"),
        "sourceManifests": {
            "noCadenceProvenance": {
                "path": str(inherited_path),
                "sha256": inherited_sha,
                "inheritance": (
                    "reuses the exact video hashes, reviewed-gap identities, split, "
                    "V5-state features, no-cadence head, and retrospective scope"
                ),
            },
            **dict(inherited.get("sourceManifests", {})),
        },
        "implementationFiles": implementation_files,
        "artifacts": {
            name: {"path": str(artifact_path), "sha256": _sha256(artifact_path)}
            for name, artifact_path in artifacts.items()
        },
        "notes": {
            "localPeaks": (
                "score-ranked non-maximum suppression uses only gap/time spacing "
                "and never re-anchors a future search window"
            ),
            "softCount": (
                "the first six outputs are unpenalized; later outputs pay an "
                "increasing logit penalty but remain eligible"
            ),
            "productionContext": (
                "a 19-input auxiliary head uses frozen production rally/dead/serve "
                "outputs, excludes raw gap duration, and contributes soft log odds "
                "rather than a hard gate"
            ),
            "suppression": "quarantined and never eligible as model input",
            "evaluationStatus": (
                "candidate-conditioned retrospective confirmation with "
                "already-opened labels, not a pristine test"
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
    parser.add_argument(
        "--no-cadence-model", type=Path, default=DEFAULT_NO_CADENCE_MODEL
    )
    parser.add_argument(
        "--no-cadence-evaluation",
        type=Path,
        default=DEFAULT_NO_CADENCE_EVALUATION,
    )
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
